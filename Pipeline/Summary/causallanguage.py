from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
import torch
from torch import nn

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

sequence = f"Hugging Face & Facebook is based in DUMBO, New York City, and"

inputs = tokenizer(sequence, return_tensors="pt")
input_ids = inputs["input_ids"]

next_token_logits = model(**inputs).logits[:, -1, :]

filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=150, top_p=1.0)

probs = nn.functional.softmax(filtered_next_token_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)

generated = torch.cat([input_ids, next_token], dim=-1)

resulting_string = tokenizer.decode(generated.tolist()[0])
print(resulting_string)
print(inputs)
print(input_ids)
print(next_token_logits)
print(filtered_next_token_logits)
print(probs)
print(next_token)
print(generated)