from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier("This is a course about the Transformer Library", candidate_labels=["eduction", "politics", "business"],)

print(result)