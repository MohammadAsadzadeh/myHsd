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
from fastapi import FastAPI, Request, Header
import subprocess
import sys
import jdatetime
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
from typing import Annotated, List, Union
import bcrypt
import jwt

access_token = os.environ.get('ACCESS_TOKEN')

myconn = mysql.connector.connect(host = "db", 
                                 user = "root",
                                 passwd =os.environ.get('DB_PASS'))
cur = myconn.cursor()
cur.execute("CREATE DATABASE IF NOT EXISTS hsddb");
cur.execute("USE hsddb")
cur.execute("CREATE TABLE IF NOT EXISTS log(id INT AUTO_INCREMENT, text VARCHAR(300) CHARACTER SET utf8mb4 COLLATE utf8mb4_persian_ci, haterate DOUBLE, threshold DOUBLE, feed BOOLEAN, ip VARCHAR(15), date VARCHAR(20), PRIMARY KEY (id))")
cur.execute("CREATE TABLE IF NOT EXISTS user(id INT AUTO_INCREMENT, username VARCHAR(100), password VARCHAR(250), salt VARCHAR(100), last_login VARCHAR(20), PRIMARY KEY (id))")
try:
    sql = "INSERT INTO user(username, password, salt) VALUES(%s, %s, %s)"
    salt = bcrypt.gensalt()
    cur.execute(sql,(os.environ.get('ADMIN_USERNAME'), bcrypt.hashpw(os.environ.get('ADMIN_PASSWORD').encode('utf-8'), salt), salt))
    myconn.commit()  
    
except Exception as e:
        print(e)  
        myconn.rollback()


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
    threshold: float

class LoginInfo(BaseModel):
    username: str
    password: str

def getTime():
    t = jdatetime.datetime.now()
    return (str)(t.year)+"/"+(str)(t.month)+"/"+(str)(t.day)+"-"+(str)(t.hour)+":"+(str)(t.minute)+":"+(str)(t.second)

app = FastAPI(title="Hate Speech Detection App",
    version="0.0.1",
    description="UT NLP Lab")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/")
async def haterating(item: TextInput, request: Request):
    item.haterate = model(item.text)
    item.date_time = getTime()
    item.feedback = None
    x_forwarded_for: str = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        client_ip = x_forwarded_for.split(",")[0]
    else:
        client_ip = request.client.host
    try:
        sql = "INSERT INTO log(text, haterate, threshold, ip, date) VALUES(%s, %s, %s, %s, %s)"
        cur.execute(sql,(item.text, item.haterate, item.threshold, client_ip, item.date_time))
        myconn.commit()  
      
    except:  
            myconn.rollback() 
    return item

@app.post("/login")
async def login(loginInfo: LoginInfo):
    print(loginInfo.username, loginInfo.password)
    try:
        sql = "SELECT password, salt FROM user WHERE username=%s"
        cur.execute(sql,[loginInfo.username])
        result = cur.fetchall()
        if(bcrypt.checkpw(loginInfo.password.encode('utf-8'), result[0][0].encode('utf-8'))):
             encoded_jwt = jwt.encode({"user": loginInfo.username},
                                       os.environ.get('SECRET_KEY'),
                                       algorithm="HS256")
             return {"access_token": encoded_jwt}
        else:
             return {"error": "Not Authorized!"}
      
    except Exception as e:  
            print(e)

@app.get("/logs")
async def getLog(feeds_only: bool, offset: int, token: Annotated[Union[List[str], None], Header()] = None):
    try:
        data = jwt.decode(token[0], os.environ.get('SECRET_KEY'), algorithms=["HS256"])
        if data["user"] != os.environ.get('ADMIN_USERNAME'):
            return {"error": "Not Authorized!"}
        records = []
        sql = "SELECT * FROM log WHERE feed IS NULL ORDER BY id LIMIT 25 OFFSET {}".format(offset*25)
        if feeds_only:
            sql = "SELECT * FROM log WHERE feed IS NOT NULL ORDER BY id LIMIT 25 OFFSET {}".format(offset*25)
        cur.execute(sql)
        result = cur.fetchall()
        for row in result:
            r = {
                "id": row[0],
                "text": row[1],
                "haterate": row[2],
                "threshold": row[3],
                "feedback": row[4],
                "ip": row[5],
                "date_time": row[6]
            }
            records.append(r)
        return records
    except Exception as e:
        print(e)
        return {"error": "Not Authorized!"}

@app.post("/feedback")
async def feedback(feed: TextInput, request: Request):
    # with open("feeds.txt", "a+") as f:
        # f.write(feed.text+"\t"+(str)(feed.haterate)+"\t"+(str)(feed.feedback)+"\t"+getTime()+"\n")
    result = False
    try:
        x_forwarded_for: str = request.headers.get("X-Forwarded-For")
        if x_forwarded_for:
            client_ip = x_forwarded_for.split(",")[0]
        else:
            client_ip = request.client.host
        sql = "INSERT INTO log(text, haterate, threshold, feed, ip, date) VALUES(%s, %s, %s, %s, %s, %s)"
        print(feed.text, feed.haterate, feed.threshold, feed.feedback, client_ip, getTime())
        print(sql)
        result = cur.execute(sql,(feed.text, feed.haterate, feed.threshold, (int)(feed.feedback), client_ip, getTime()))
        myconn.commit()  
    
    except Exception as e:
        print(e)  
        myconn.rollback()
    return {"feed": feed, "result":result}






