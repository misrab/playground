# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt

# import requests
# import pandas as pd
# from datetime import datetime, timedelta

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# model_name = "google/flan-t5-base"
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, force_download=True, resume_download=False)
# model = AutoModel.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, force_download=True, resume_download=False)


# input_text = "translate English to German: How old are you?"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids
# print(input_ids)
# outputs = model.generate(input_ids, max_new_tokens=100)
# print(tokenizer.decode(outputs[0]))


# write a loop for user input
while True:
    input_text = input("Enter text to translate: ")
    if input_text == "quit":
        break
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(outputs[0]))