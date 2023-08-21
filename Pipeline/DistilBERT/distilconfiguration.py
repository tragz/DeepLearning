from transformers import DistilBertModel, DistilBertConfig

# Initializing a DistilBERT configuration
configuration = DistilBertConfig()
print(configuration)
print(type(configuration))
print()

model = DistilBertModel(configuration)
print(model)
print(type(model))
print()

configuration = model.config
print(configuration)
print(type(configuration))
print()