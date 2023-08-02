import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from string import punctuation
import transformers
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForTokenClassification
from hazm import *
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from models import RuleBasedModel, BertBasedModel, PiplineModel
from utils import Preprocessor
from string import punctuation
import getopt, sys
from PIL import Image
import os
import torch
from torch import nn
import arabic_reshaper
from bidi.algorithm import get_display
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
import subprocess
import sys
import jdatetime
from fastapi.middleware.cors import CORSMiddleware



model_exist = False
for f in os.listdir():
    if f == "hatespeech_bert_model.pth":
        model_exist = True
        break
if not model_exist:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

    url = "https://drive.google.com/uc?id=1v0KVzMvlxEh3dPP1f9lFTTxhvUA-jKOv"
    output = "hatespeech_bert_model.pth"
    gdown.download(url, output, quiet=False)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

#BERT MODEL
bert_tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")


bert_model = BertBasedModel("HooshvareLab/bert-base-parsbert-uncased")

bert_model.load_state_dict(torch.load('hatespeech_bert_model.pth',map_location=device))
bert_model = bert_model.to(device)

#NER MODEL
normalizer = Normalizer(persian_numbers=False)

ner_model_name = 'HooshvareLab/bert-base-parsbert-armanner-uncased'
ner_config = AutoConfig.from_pretrained(ner_model_name)
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
ner_labels = list(ner_config.label2id.keys())

#RULEBASED MODEL
offensive_df = pd.read_excel('Data/dictionary/Offensive_Dictionary.xlsx',
                               sheet_name="Offensive")

ambiguous_df = pd.read_excel('Data/dictionary/Offensive_Dictionary.xlsx',
                               sheet_name="Ambiguous")

preprocessor = Preprocessor(normalizer, punctuation)
rulebased_model = RuleBasedModel(offensive_df, ambiguous_df, ner_tokenizer, ner_model, ner_labels, preprocessor)

model = PiplineModel(preprocessor, rulebased_model, bert_model, bert_tokenizer, device)

class TextInput(BaseModel):
    text: str
    haterate: float
    date_time: str
    feedback: bool

def getTime():
    t = jdatetime.datetime.now()
    return (str)(t.year)+"/"+(str)(t.month)+"/"+(str)(t.day)+"-"+(str)(t.hour)+":"+(str)(t.minute)+":"+(str)(t.second)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/")
async def haterating(item: TextInput):
    item.haterate = model(item.text)
    item.date_time = getTime()
    item.feedback = None
    return item

@app.post("/feedback")
async def feedback(feed: TextInput):
    with open("feeds.txt", "a+") as f:
        f.write(feed.text+"\t"+(str)(feed.haterate)+"\t"+(str)(feed.feedback)+"\t"+getTime()+"\n")
    return feed






