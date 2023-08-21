from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


seq_a = "This is short sentence"
seq_b = "This is a rather long sequence. It is at least longer than the seq A."

enc_seq_a = tokenizer(seq_a)["input_ids"]
enc_seq_b = tokenizer(seq_b)["input_ids"]
print("Input Ids")
print(enc_seq_a)
print(enc_seq_b)

print("Tokenizer")
print(tokenizer(seq_a))
print(tokenizer(seq_b))

print("Tokens")
print(tokenizer.tokenize(seq_a))
print(tokenizer.tokenize(seq_b))

print("Length")
print(len(enc_seq_a), len(enc_seq_b))

# Batching
# first sequence needs to be padded up to the length of the second one
# OR second sequence needs to be truncated down to length of the first one.

pad_seq = tokenizer([seq_a, seq_b], padding=True)
print(pad_seq)
print(pad_seq["input_ids"])
print(pad_seq["attention_mask"])
