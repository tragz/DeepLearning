from torch import nn
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(inputs)
print(tokenizer.tokenize("We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."))
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
print(pt_batch)
print()
for key, value in pt_batch.items():
    print(f"{key}: {value.numpy().tolist()}")
pt_outputs = pt_model(**pt_batch)
print(pt_outputs)
pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(pt_predictions)

pt_outputs = pt_model(**pt_batch, labels = torch.tensor([1, 0]))
print(pt_outputs)