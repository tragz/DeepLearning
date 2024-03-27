from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-cased")

sequence = "Distilled models are smaller that the models they mimic. Using them instead of the large " \
    f"versions would help {tokenizer.mask_token} our carbon footprint."

inputs = tokenizer(sequence, return_tensors="pt")
mask_token_index = torch.where(tokenizer.mask_token_id == inputs["input_ids"])[1]

token_logits = model(**inputs).logits
mask_token_index = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(mask_token_index, 15, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))