
from waitress import serve
from flask import Flask, render_template, request, redirect,url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)


chat = Flask(__name__)
chat.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(chat)
chat.config["DEBUG"] = True



class Todo(db.Model):
    product_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False)
    email=db.Column(db.String(20), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"Todo('{self.product_id}', '{self.name}','{self.email}','{self.date_created}'  )"



input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

@chat.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        task_content1 = request.form.get('name')
        task_content2 = request.form.get('email')
        new_task = Todo(name=task_content1,email=task_content2)

        try:
            db.session.add(new_task)
            db.session.commit()
            return redirect( url_for('home') )
        except:
            return 'There is an error'

    else:
        return render_template('login.html')



@chat.route("/home")
def home():
    return render_template("home.html")


@chat.route("/get")
#function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    sentence = userText
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
               if tag == intent["tag"]:
                   return random.choice(intent['responses'])
    else:
        return "I don't understand"


@chat.route("/done")
def done():
    return render_template("done.html")


if __name__ == "__main__":
    serve(chat)


