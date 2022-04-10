from engine import training
import csv
import pickle
import pandas as pd
import gridfs
import os
from pymongo import MongoClient

import bcrypt
import re
import nltk


def mongo_conn():
    try:
        conn = MongoClient(host='127.0.0.1', port=27017)
        print('mongo connected')
        return conn
    except:
        print('error in mongo connenction')

conn = mongo_conn()

db = conn.sandeep_db

with open('finalized_model.pickle', 'rb') as handle:
    b = pickle.load(handle)










# df = pd.read_csv('input/IMDB Dataset.csv')
# coloms = df.columns
# df['text'] = df[coloms[0]]
# df['label'] = df[coloms[1]]

# trainer = training(df)

# trainer.train_model()