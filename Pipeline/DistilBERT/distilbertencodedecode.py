from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(("bert-base-cased"))
seq_a = "HuggingFace is based in NYC"
seq_b = "Where is HuggingFaced based?"
seq_c = "Raghav you have used HuggingFaced BertTokenizer"

enc_dict = tokenizer(seq_a, seq_b)
dec = tokenizer.decode(enc_dict["input_ids"])

print(enc_dict)
print(dec)