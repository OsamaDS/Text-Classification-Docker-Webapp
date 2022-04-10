from base64 import encode
from sklearn import preprocessing, naive_bayes, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn import decomposition, ensemble
import nltk
import re
from nltk import WordNetLemmatizer
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')
import pandas as pd
import numpy as np
import string
import pickle

class training:
    def __init__(self, df):
        self.lemma_ = WordNetLemmatizer()
        self.df = df
        self.df.dropna(inplace=True)
        self.df.drop_duplicates('text',inplace=True)
        self.X = self.df['text']
        self.y = self.df['label']
    
    def data_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)
        print('--------------------------')
        print('Data is splitted')
        print('--------------------------')
        encoder = preprocessing.LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.fit_transform(y_test)

        return X_train, X_test, y_train, y_test, encoder

    def clean_text(self, text):
        text = str(text)
        text = re.sub('[^a-zA-Z]', " ", text) #remove punctuations and numbers
        text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text) # Single character removal
        text = re.sub(r'\s+', " ", text) #remove extra spaces
        text = text.replace("ain't", "am not").replace("aren't", "are not")
        text = ' '.join(tex.lower() for tex in text.split(' ')) # Lowering cases
        sw = nltk.corpus.stopwords.words('english')
        text = ' '.join(tex for tex in text.split() if tex not in sw) #removing stopwords
        text = ' '.join(self.lemma_.lemmatize(x) for x in text.split()) #lemmatization
        
        return text

    def vectorizer(self):
        X_train, X_test, y_train, y_test, encoder = self.data_split()
        
        X_train = X_train.apply(self.clean_text)
        X_test = X_test.apply(self.clean_text)
        print('--------------------------')
        print('Text cleaning is done!!!!!!!!!!')
        print('--------------------------')
        X_train = list(X_train)
        X_test = list(X_test)

        vec = TfidfVectorizer(max_features=5000)
        print('--------------------------')
        print('getting feature vectors')
        print('--------------------------')
        vec.fit(X_train)
        print('--------------------------')
        print('transforming feature vectors')
        print('--------------------------')
        # pickle.dump(vec, open('vectorizer.pkl', 'wb'))
        train_vectors = vec.transform(X_train)
        test_vectors = vec.transform(X_test)

        return train_vectors, test_vectors, y_train, y_test, vec, encoder

    def train_model(self):
        nb_model = naive_bayes.GaussianNB()
        train_vectors, test_vectors, y_train, y_test, vec, encoder = self.vectorizer()
        print('--------------------------')
        print('model trainig started')
        print('--------------------------')
        nb_model.fit(train_vectors.toarray(), y_train)
        # print('--------------------------')
        # pickle.dump(nb_model, open('model.pkl', 'wb'))
        # print('model saved!!!!!')
        y_pred = nb_model.predict(test_vectors.toarray())
        print(classification_report(y_test, y_pred))

        return nb_model, vec, encoder

