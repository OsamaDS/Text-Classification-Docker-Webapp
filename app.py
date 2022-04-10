from importlib_metadata import metadata
from engine import training
import csv
import pickle
import pandas as pd
from gridfs import GridFS
from flask import Flask, render_template, url_for, request, session, redirect
import os
from flask_pymongo import PyMongo
import bcrypt
import re
import nltk

app = Flask(__name__)
app.secret_key = "testing"
app.config["SECRET_KEY"]
app.config['MONGO_URI'] = 'mongodb://localhost:27017/sandeep_db'
app.config['FILE_UPLOADS'] = "C:\\Users\\Osama\\Downloads\\Sandeep Project\\App\\static\\files"
mongo = PyMongo(app)

#client = pymongo.MongoClient("mongodb://localhost:27017")
#db = client['']


#records = db.register

model_fs = GridFS(mongo.db, collection='models')
vec_fs = GridFS(mongo.db, collection='vectorizer')
enc_fs = GridFS(mongo.db, collection='encoder')


@app.route('/')
def index():
    # if 'username' in session:
    #     return 'You are logged in as ' + session['username']

    return render_template('signin.html')

@app.route('/login', methods=['POST'])
def login_user():
    user_collection = mongo.db.register_users
    
    if request.method=='POST':
        username = request.form['username']
        password = request.form['password']
        login_user = user_collection.find_one({'username' : username})

        if login_user['username'] == "admin" and login_user['password'] == "asdf":
            return redirect(url_for('index'))

        if login_user:
            if login_user['password'] == password:
                session['username'] = username
                return redirect(url_for('models'))               
            else:
                return ("please enter valid password")

        return 'Invalid username/password combination'

@app.route('/auth/login')
def login():
    
    return render_template('signin.html')

@app.route('/users', methods=['GET','POST'])
def show_users():
    user_collection = mongo.db.register_users
    if request.method == 'POST':
        user = request.form["username"]
        pwd = request.form["password"]
        confirm_pass = request.form["confirm_password"]
        existing_user = user_collection.find_one({'username' : user})
        if existing_user is None:
            #hashpass = bcrypt.hashpw(pwd.encode('utf-8'), bcrypt.gensalt())
            #CP_hashpass = bcrypt.hashpw(confirm_pass.encode('utf-8'), bcrypt.gensalt())
            user_collection.insert_one({'username' : user, 'password' : pwd, 'confirm_password':confirm_pass})
            session['username'] = request.form['username']
            return "user registered"
        else:
            return "user already exists"


@app.route('/auth/register', methods=['GET', 'POST'])
def signup():
    
    return render_template('signup.html')

@app.route('/upload')
def getfile():
    return render_template('upload.html')

@app.route('/datafile', methods=['GET','POST'])
def uploadfile():
    data = []
    model_collection = mongo.db['models']
    #model_collection = mongo.db['vectorizers']
    if request.method == 'POST':
        if request.files:
            uploaded_file = request.files['csvfile'] 
            model_name = request.form['modelname']
            df = pd.read_csv(uploaded_file)
            coloms = df.columns
            df['text'] = df[coloms[0]]
            df['label'] = df[coloms[1]]

            #df = df.loc[:100,:]

            trainer = training(df)

            model, vectorizer, encoder = trainer.train_model()
            # pickle.dump(model, open('modelstate.pickle', 'wb'))
            # pickle.dump(vectorizer, open('vecstate.pickle', 'wb'))
            # pickle.dump(encoder, open('encstate.pickle', 'wb'))
            pickle_model = pickle.dumps(model)
            pickle_vectorizer = pickle.dumps(vectorizer)
            pickle_encoder = pickle.dumps(encoder)

            model_fs.put(pickle_model, filename=session['username'], metadata={'modelname':model_name})
            vec_fs.put(pickle_vectorizer, filename=session['username'], metadata={'modelname':model_name})
            enc_fs.put(pickle_encoder, filename=session['username'], metadata={'modelname':model_name})
                                                            

            # model_collection.insert_one(
            #     {
            #         "username": session['username'],
            #         "model_file": pickle_model,
            #         "model_name": model_name,
            #         "model_vectorizer": pickle_vectorizer,
            #         "label_encoder": pickle_encoder
            #     }
            # )

            tmp = df['label'].value_counts()
            x = list(tmp.index)
            y = list(tmp.values)

            df['text_len'] = df['text'].apply(lambda x: len(str(x).split()))
            tmp2 = df['text_len'].value_counts()
            x2 = list(tmp2.index)
            y2 = list(tmp2.values)

            return render_template('charts.html', x=x, y=y, x2=x2, y2=y2)

@app.route('/deploy_upload', methods=['GET','POST'])
def deploy_upload():
    #model_collection = mongo.db['models']
    
    data = model_fs.find({"filename":session['username']})
    print('fs_data = ', data)

    model_names = []
    for d in data:
        i = d.metadata
        print('meta:', i['modelname'])
        model_names.append(i['modelname'])
    
    #user_document = model_collection.find({'username': session['username']})
    # for i in user_document:
    #     print('hushhh')
    #     model_names.append(i['model_name'])
    #     print(i['model_name'])
    #model_names.append('usama')
    len_ = len(model_names)
    print(model_names)
    return render_template('deploy_upload.html', len_=len_, values=model_names)

@app.route('/deploy', methods=['GET','POST'])
def deployfile():
    model_collection = mongo.db['models']
    model_names = []
    if request.method == 'POST':
        if request.files:
            uploaded_file = request.files['csvfile']   
            model_name = request.form['btn']
            df = pd.read_csv(uploaded_file)
            df.dropna(inplace=True)
            coloms = df.columns
            df['text'] = df[coloms[0]]
            #df['label'] = df[coloms[1]]

            user_model = model_fs.find({"filename":session['username']})
            user_vec = vec_fs.find({"filename":session['username']})
            user_enc = enc_fs.find({"filename":session['username']})
            print('fs_data = ', user_model)
            # model_file = None
            # label_enc = None
            # vect = None
            for i in user_model:
                d = i.metadata
                if d['modelname'] == model_name:
                    print('d======')
                    model_file = i.read()
            for i in user_vec:
                d = i.metadata
                if d['modelname'] == model_name:
                    print('d======')
                    vect = i.read()
            for i in user_enc:
                d = i.metadata
                if d['modelname'] == model_name:
                    print('d======')
                    label_enc = i.read()


            # user_model = model_collection.find_one({'username': session['username'], 'model_name': model_name})
            # model_file = user_model['model_file']
            # vect = user_model['model_vectorizer']
            # label_enc = user_model['label_encoder']
            
            model = pickle.loads(model_file)
            vec = pickle.loads(vect)
            encoder = pickle.loads(label_enc)

            #print(user_model['model_name'])
            res, df_new = modelPredict(model, vec, encoder, df)
            df_new.to_csv('static/files/results2.csv')

            return redirect(url_for('model_result'))
            
@app.route('/model_results', methods=['POST', 'GET'])
def model_result():
    return render_template('download.html')






@app.route('/users/models')
def models():
    return render_template('models.html')

@app.route('/users/templates')
def templates():
    return render_template('templates.html')

@app.route('/charts')
def chart():
    return render_template('charts.html')


def modelPredict(model, vec, encoder, df):
    #df = pd.read_csv('static/files/IMDB Dataset.csv')
    #df = df.loc[5:15,:]
    texts = list(df['text'])
    print('len of text: ',len(texts))
    result = []
    text_input = []
    for text in texts:
        text_input.append(text)
        text = str(text)
        text = re.sub('[^a-zA-Z]', " ", text) #remove punctuations and numbers
        text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text) # Single character removal
        text = re.sub(r'\s+', " ", text) #remove extra spaces
        text = text.replace("ain't", "am not").replace("aren't", "are not")
        text = ' '.join(tex.lower() for tex in text.split(' ')) # Lowering cases
        #sw = nltk.corpus.stopwords.words('english')
        #text = ' '.join(tex for tex in text.split() if tex not in sw) #removing stopwords
        #text = ' '.join(self.lemma_.lemmatize(x) for x in text.split()) #lemmatization
        #print('text: ',text)
        text = [text]
        vector = vec.transform(text)
        res = model.predict(vector.toarray())
        res = encoder.inverse_transform(res)
        result.append(res)
        #text_input.append(inp_text)
        #print(res)
    data = {'text':text_input, "predictions":result}
    df = pd.DataFrame(data)
    
    return result, df

if __name__ == '__main__':
    app.secret_key = 'mysecret'
    app.run(debug=True)
