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
app = FastAPI()
@app.post("/")
async def haterating(item: TextInput):
    item.haterate = model(item.text)
    return item





