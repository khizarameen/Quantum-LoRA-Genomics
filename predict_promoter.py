from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Saved model aur tokenizer load karein
model_path = "./my_fine_tuned_dnabert"
print("Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
model.eval()  # Evaluation mode

# 2. Koi bhi test DNA sequence (misal ke tor par 251 bp ka sequence)
test_seq = "AGCTCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"

# 3. Tokenize karein
inputs = tokenizer(test_seq, return_tensors="pt", padding="max_length", truncation=True, max_length=251)

# 4. Prediction karein
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

# 5. Result print karein
print(f"Prediction probabilities: {probabilities.tolist()}")
if predicted_class == 1:
    print("Result: Yeh sequence ek **Promoter** hai!")
else:
    print("Result: Yeh sequence **Non-Promoter** hai.")
