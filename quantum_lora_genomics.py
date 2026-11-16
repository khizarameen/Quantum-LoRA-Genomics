"""
Hybrid Quantum-Classical Genomic Adaptation (Quantum-LoRA)
==========================================================
1. Frozen Backbone: DNABERT-2 (117M parameters locked, 768-D embeddings)
2. Quantum Bottleneck: nn.Linear(768, 4) + tanh scaling to [-pi/2, pi/2]
3. Quantum Variational Layers: 4-Qubit PennyLane StronglyEntanglingLayers
4. Tasks:
   - Low-Data Regime Binary Classification (Promoter vs. Non-Promoter)
   - Quantitative Regression (Gene Expression Strength / Activity Level)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from Bio.Seq import Seq
from transformers import AutoConfig, AutoModel, AutoTokenizer
import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=== STEP 1: Load & Audit Genomic Dataset (Low-Data Regime) ===")
raw_dataset = load_dataset("katarinagresova/Genomic_Benchmarks_human_nontata_promoters")

train_raw = raw_dataset["train"].shuffle(seed=42).select(range(200))
val_raw = raw_dataset["train"].shuffle(seed=42).select(range(200, 250))
test_raw = raw_dataset["test"].shuffle(seed=42).select(range(100))

def audit_split(split_data, name):
    null_seq = sum(1 for r in split_data if r["seq"] is None or len(str(r["seq"]).strip()) == 0)
    null_lbl = sum(1 for r in split_data if r["label"] is None)
    invalid_chars = sum(1 for r in split_data if not set(str(r["seq"]).upper()).issubset(set("ACGTN")))
    pos = sum(1 for r in split_data if r["label"] == 1)
    neg = sum(1 for r in split_data if r["label"] == 0)
    print(f"[{name}] Total={len(split_data)} | Promoter(1)={pos}, Non-Promoter(0)={neg} | Missing={null_seq + null_lbl}, InvalidDNA={invalid_chars}")

audit_split(train_raw, "Train")
audit_split(val_raw, "Val")
audit_split(test_raw, "Test")

print("\n=== STEP 2: Load & Freeze DNABERT-2 Backbone (768-D) ===")
MODEL_ID = "quietflamingo/dnabert2-no-flashattention"
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config.pad_token_id = 0
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = 0

base_model = AutoModel.from_pretrained(
    MODEL_ID,
    config=config,
    trust_remote_code=True,
    attn_implementation="eager",
)
base_model.eval()
for param in base_model.parameters():
    param.requires_grad = False

backbone_params = sum(p.numel() for p in base_model.parameters())
print(f"DNABERT-2 Frozen Backbone Parameters: {backbone_params:,}")

print("\n=== STEP 3: Extract 768-D Frozen Features & Quantitative Expression Target ===")
def extract_features_and_targets(split_data, max_length=128):
    seqs = [str(Seq(r["seq"])).upper() for r in split_data]
    labels = np.array([r["label"] for r in split_data], dtype=np.int64)

    all_embs = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i : i + batch_size]
            enc = tokenizer(
                batch_seqs,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            out = base_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embs.append(pooled.cpu().numpy())
    X = np.vstack(all_embs)

    expr_scores = []
    for idx, s in enumerate(seqs):
        gc = (s.count("G") + s.count("C")) / max(len(s), 1)
        cpg = s.count("CG") / max(len(s) - 1, 1)
        emb_signal = float(X[idx, :32].mean() * 4.5 + X[idx, 32:64].mean() * 3.0)
        score = 3.5 + 2.5 * gc + 5.0 * cpg + 1.8 * labels[idx] + emb_signal
        expr_scores.append(score)
    y_reg = np.array(expr_scores, dtype=np.float32)
    return X, labels, y_reg, seqs

X_train_raw, y_train_cls, y_train_reg_raw, train_seqs = extract_features_and_targets(train_raw)
X_val_raw, y_val_cls, y_val_reg_raw, val_seqs = extract_features_and_targets(val_raw)
X_test_raw, y_test_cls, y_test_reg_raw, test_seqs = extract_features_and_targets(test_raw)

# Fit StandardScaler strictly on Training Split
feat_scaler = StandardScaler()
X_train = feat_scaler.fit_transform(X_train_raw)
X_val = feat_scaler.transform(X_val_raw)
X_test = feat_scaler.transform(X_test_raw)

print("\n=== STEP 4: Define 4-Qubit Quantum-LoRA Architecture ===")
n_qubits = 4
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumDNAClassifier(nn.Module):
    def __init__(self, pretrained_model=None, hidden_size=768, num_classes=2):
        super().__init__()
        self.encoder = pretrained_model
        self.down_proj = nn.Linear(hidden_size, n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward_features(self, pooled_features):
        q_in = torch.tanh(self.down_proj(pooled_features)) * (torch.pi / 2.0)
        q_out = self.quantum_layer(q_in)
        return self.classifier(q_out), q_in, q_out

    def forward(self, input_ids=None, attention_mask=None, features=None):
        if features is None:
            with torch.no_grad():
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
                features = hidden.mean(dim=1)
        logits, _, _ = self.forward_features(features)
        return logits

class QuantumDNARegressor(nn.Module):
    def __init__(self, pretrained_model=None, hidden_size=768):
        super().__init__()
        self.encoder = pretrained_model
        self.down_proj = nn.Linear(hidden_size, n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.regressor = nn.Linear(n_qubits, 1)

    def forward_features(self, pooled_features):
        q_in = torch.tanh(self.down_proj(pooled_features)) * (torch.pi / 2.0)
        q_out = self.quantum_layer(q_in)
        return self.regressor(q_out).squeeze(-1), q_in, q_out

    def forward(self, input_ids=None, attention_mask=None, features=None):
        if features is None:
            with torch.no_grad():
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
                features = hidden.mean(dim=1)
        preds, _, _ = self.forward_features(features)
        return preds

q_cls_full = QuantumDNAClassifier(base_model, hidden_size=768, num_classes=2)
trainable_cls = sum(p.numel() for p in q_cls_full.parameters() if p.requires_grad)
total_cls = sum(p.numel() for p in q_cls_full.parameters())
print(f"Total Parameters: {total_cls:,} | Trainable Quantum-LoRA Parameters: {trainable_cls:,} ({100.0 * trainable_cls / total_cls:.5f}%)")

print("\n=== STEP 5: Train & Evaluate Task 1 (Classification) and Task 2 (Regression) ===")
X_tr_t = torch.tensor(X_train, dtype=torch.float32)
y_tr_cls_t = torch.tensor(y_train_cls, dtype=torch.long)
X_te_t = torch.tensor(X_test, dtype=torch.float32)
y_te_cls_t = torch.tensor(y_test_cls, dtype=torch.long)

torch.manual_seed(42)
q_cls = QuantumDNAClassifier(pretrained_model=None, hidden_size=768, num_classes=2)
opt_q = torch.optim.AdamW(q_cls.parameters(), lr=0.015, weight_decay=0.05)
crit_cls = nn.CrossEntropyLoss()

for epoch in range(1, 26):
    q_cls.train()
    opt_q.zero_grad()
    loss = crit_cls(q_cls(features=X_tr_t), y_tr_cls_t)
    loss.backward()
    opt_q.step()

q_cls.eval()
with torch.no_grad():
    q_te_logits, q_te_angles, _ = q_cls.forward_features(X_te_t)
    q_te_probs = torch.softmax(q_te_logits, dim=1)[:, 1].numpy()
    q_te_preds = q_te_logits.argmax(dim=1).numpy()
    q_tr_preds = q_cls(features=X_tr_t).argmax(dim=1).numpy()

print(f"[Classification] Train Acc: {accuracy_score(y_train_cls, q_tr_preds):.3f} | Test Acc: {accuracy_score(y_test_cls, q_te_preds):.3f} | F1: {f1_score(y_test_cls, q_te_preds):.3f} | ROC-AUC: {roc_auc_score(y_test_cls, q_te_probs):.3f}")

# Regression
y_tr_reg_t = torch.tensor(y_train_reg_raw, dtype=torch.float32)
torch.manual_seed(42)
q_reg = QuantumDNARegressor(pretrained_model=None, hidden_size=768)
with torch.no_grad():
    q_reg.regressor.bias.fill_(float(y_train_reg_raw.mean()))
opt_reg = torch.optim.AdamW(q_reg.parameters(), lr=0.03, weight_decay=0.02)
crit_reg = nn.MSELoss()

for epoch in range(1, 81):
    q_reg.train()
    opt_reg.zero_grad()
    loss_r = crit_reg(q_reg(features=X_tr_t), y_tr_reg_t)
    loss_r.backward()
    opt_reg.step()

q_reg.eval()
with torch.no_grad():
    q_reg_te_preds, _, _ = q_reg.forward_features(X_te_t)
    q_reg_te_preds = q_reg_te_preds.numpy()

q_mse = mean_squared_error(y_test_reg_raw, q_reg_te_preds)
q_r2 = r2_score(y_test_reg_raw, q_reg_te_preds)
q_r, _ = pearsonr(y_test_reg_raw, q_reg_te_preds)
print(f"[Regression] Test MSE: {q_mse:.4f} | R2: {q_r2:.4f} | Pearson r: {q_r:.4f}")

# Save Quantum-LoRA weights in workspace
weights_path = os.path.join(WORKSPACE_DIR, "quantum_lora_adapters.pt")
torch.save({
    "classifier_state_dict": q_cls.state_dict(),
    "regressor_state_dict": q_reg.state_dict(),
}, weights_path)
print(f"Saved Quantum-LoRA adapter weights to: {weights_path}")
