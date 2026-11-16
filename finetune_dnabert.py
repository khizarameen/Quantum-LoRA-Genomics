import os
import sys

# Ensure clean imports in case transformers/huggingface_hub were updated
for _mod in [k for k in list(sys.modules) if k.startswith("transformers") or k.startswith("huggingface_hub")]:
    del sys.modules[_mod]

from datasets import load_dataset
import transformers.utils.import_utils
transformers.utils.import_utils._accelerate_available = True

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import transformers.training_args
import transformers.trainer
transformers.training_args.is_accelerate_available = lambda *args, **kwargs: True
transformers.trainer.is_accelerate_available = lambda *args, **kwargs: True

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 1. Dataset Load Karein
print("1. Loading dataset...")
raw_dataset = load_dataset("katarinagresova/Genomic_Benchmarks_human_nontata_promoters")

# Low-Data Regime subset taake CPU par jaldi aur saaf training nazar aaye
train_subset = raw_dataset["train"].shuffle(seed=42).select(range(128))
test_subset = raw_dataset["test"].shuffle(seed=42).select(range(64))

# 2. Tokenizer Load Karein (DNABERT-2)
model_name = "zhihan1996/DNABERT-2-117M"
print(f"2. Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = 0

# 3. Tokenization Function
def tokenize_function(examples):
    # Dataset mein column ka naam 'seq' hai
    seq_col = "seq" if "seq" in examples else "sequence"
    return tokenizer(
        examples[seq_col],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

print("3. Tokenizing dataset...")
encoded_train = train_subset.map(tokenize_function, batched=True)
encoded_eval = test_subset.map(tokenize_function, batched=True)

# 4. Classification Model Load Karein (num_labels = 2 kyunki Promoter vs Non-Promoter hai)
print("4. Loading DNABERT-2 model for Sequence Classification...")
config = AutoConfig.from_pretrained(model_name, num_labels=2, trust_remote_code=True)
config.pad_token_id = 0

# Ensure cached bert_layers.py is patched for PyTorch meta device compatibility
import glob
for _bf in glob.glob(os.path.expanduser(r"~\.cache\huggingface\**\bert_layers.py"), recursive=True):
    with open(_bf, "r", encoding="utf-8") as _f:
        _txt = _f.read()
    if "torch.Tensor(_get_alibi_head_slopes" in _txt or "self.alibi = self.alibi.to(hidden_states.device)" in _txt:
        _txt = _txt.replace(
            "slopes = torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device)",
            "slopes = torch.tensor(_get_alibi_head_slopes(n_heads), dtype=torch.float32, device=relative_position.device)",
        ).replace(
            "self.alibi = self.alibi.to(hidden_states.device)",
            "self.rebuild_alibi_tensor(size=self._current_alibi_size, device=hidden_states.device)",
        ).replace(
            "assert isinstance(self.token_type_ids, torch.LongTensor)",
            "if self.token_type_ids.device != (input_ids.device if input_ids is not None else inputs_embeds.device): self.token_type_ids = torch.zeros(self.token_type_ids.shape, dtype=torch.long, device=(input_ids.device if input_ids is not None else inputs_embeds.device))",
        )
        with open(_bf, "w", encoding="utf-8") as _f:
            _f.write(_txt)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config,
    trust_remote_code=True,
)

# PEFT / Fast Fine-Tuning: Backbone freeze kar ke pooler + classifier train karein
for name, param in model.named_parameters():
    if "classifier" not in name and "pooler" not in name:
        param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,} | Trainable Parameters: {trainable_params:,}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

# 5. Training Arguments Set Karein (Workspace ke andar save hoga, venv ke bahar)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dnabert_promoters_output")
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_fine_tuned_dnabert")

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",            # Har epoch ke baad evaluation hogi
    save_strategy="epoch",
    learning_rate=5e-4,               # Fine-tuning learning rate
    per_device_train_batch_size=8,    # Agar GPU memory kam ho toh ise 4 ya 2 kar dein
    per_device_eval_batch_size=8,
    num_train_epochs=3,               # 3 epochs is task ke liye kafi hain
    weight_decay=0.01,
    logging_steps=4,
    load_best_model_at_end=True,
    report_to="none",
)

# 6. Trainer Initialize Karein
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_eval,
    compute_metrics=compute_metrics,
)

# 7. Training Start Karein
print("\n5. Starting Fine-Tuning...")
train_result = trainer.train()
eval_metrics = trainer.evaluate()
print("\n6. Final Evaluation Metrics:", eval_metrics)

# 8. Model ko Workspace Folder mein Save Karein (venv ke bahar)
print(f"\n7. Saving fine-tuned model and tokenizer to: {save_dir}")
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print("Training completed and model saved successfully at './my_fine_tuned_dnabert'!")
