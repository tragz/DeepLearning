from transformers import pipeline
from pprint import pprint

unmasker = pipeline("fill-mask")
pprint(unmasker(f"HuggingFace is creating a {unmasker.tokenizer.mask_token} that the community uses to NLP tasks."))