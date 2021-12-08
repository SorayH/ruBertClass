# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import json
from Class_neur import bert_func0
from Class_neur import bert_func1
import sqlite3
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from Class_neur import BERTClass

nltk.download('stopwords')

def preprocess_text(text):

    russian_stopwords = stopwords.words("russian")
    tokens = text.split()
    res = []

    for token in tokens:
        if ((token.lower() not in russian_stopwords) and (token != " ") and (token.strip() not in punctuation)):
            res.append(token)

    text = " ".join(res).replace("????", "")

    return text

def post_processed_text(text):

    text = json.dumps(text, ensure_ascii=False)

    return text

def neur_call(neur_type, text):

    text = preprocess_text(text)
    x = eval(neur_type)()
    res = x.predict(text)
    return res

