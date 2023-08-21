from transformers import DistilBertTokenizer, DistilBertModel
import torch


tokenizers = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

inputs = tokenizers("Hello, my dog is cute", return_tensors='pt')
outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state

print(f"Input :\n{inputs}")
print(f"Output:\n{outputs}")
print(f"Last Hidden State:\n{last_hidden_state}")