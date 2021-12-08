# -*- coding: utf-8 -*-
from flask import Flask
from flask_restful import Api, Resource
from main import neur_call
from db import res_db
from Class_neur import BERTClass

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)

neurs = res_db('neur')

class Predict(Resource):

    def get(self, neur_type, text):

        for neur in neurs:
            if neur_type == neur:
                res = neur_call(neur_type, text)
                return res, 200

        return "Quote not found", 404

api.add_resource(Predict, "/predict/<neur_type>/<text>")
if __name__ == '__main__':
    app.run(debug=True)