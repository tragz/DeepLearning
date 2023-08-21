from transformers import DistilBertTokenizer, DistilBertForMultipleChoice
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
model = DistilBertForMultipleChoice.from_pretrained('distilbert-base-cased')

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."
labels = torch.tensor(0).unsqueeze(0)
print(labels)

encoding = tokenizer([[prompt, choice0], [prompt, choice1]], return_tensors='pt', padding=True)
outputs = model(**{k:v.unsqueeze(0) for k,v in encoding.items()}, labels=labels)

loss = outputs.loss
logits = outputs.logits

print(loss)
print(logits)
print(encoding)
print(outputs)

