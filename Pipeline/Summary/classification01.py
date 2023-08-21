from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("I Hate you")[0]
print(f"label : {result['label']}, with score: {round(result['score'], 4)}")

result = classifier("i love you")[0]
print(f"label : {result['label']}, with score: {round(result['score'], 4)}")