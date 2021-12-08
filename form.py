#- * - coding: utf - 8 - *-
from flask import Flask, render_template, request
import json
from main import bert_func0
from db import res_db
from main import neur_call
import json
import sqlite3
from Class_neur import BERTClass

app = Flask(__name__)

neurs = res_db('neur_type')

@app.route('/', methods=['post', 'get'])
def form():
    res = ''

    if request.method == 'POST':

        text = request.form.get('text')  # запрос к данным формы
        neur_type = request.form.get('neur_type')
        res = neur_call(neur_type, text)

    if res == '':
        return render_template('out.html', neur=neurs)  # !!! как не перезаписывать страницу
    else:
        return render_template('out.html', neur=neurs, res = res)  # !!! как не перезаписывать страницу

if __name__ == "__main__":
    app.run()
