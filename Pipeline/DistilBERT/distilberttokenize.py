from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
embeddings  = tokenizer.tokenize("I have a new GPU!")
print(embeddings)

from transformers import XLNetTokenizer
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
embeddings = tokenizer.tokenize("Don't your love 🤗 Transformers? We sure do.")
print(embeddings)