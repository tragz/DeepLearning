

# Pipeline funciton in transformer library
# Pre-processing -> Model -> Post-Processing
# https://huggingface.co/transformers/v4.10.1/main_classes/pipelines.html#transformers.pipeline

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
#No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b
# DistilBERT - distilled version of BERT - smaller, faster, cheaper and lighter Transformer model
# 40% less parameters
print(classifier("I've been waiting for a Huggingface course my whole life"))
print(classifier('We are very happy to show you the 🤗 Transformers library.'))

results = classifier(["We are very happy to show you the 🤗 Transformers library.",
                      "We hope you don't hate it."])
for result in results:
    print(result)
    #print(f"label: {result['label']}, with score: {round(result['score'], 4)}")




