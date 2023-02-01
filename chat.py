import random
import json
import torch
from scrape_for_answer import search
from model import SeqClassifier, bag_of_words
import nltk


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = SeqClassifier(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def predict(sentence):
    sentence = nltk.word_tokenize(sentence)
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
                if intent["tag"] == "science":
                    try:
                        result = search(sentence)
                    except:
                        result = random.choice(intent['responses'])
                else:
                    result = random.choice(intent['responses'])
    else:
        result = "Sorry, I do Not Understand!"
    return result

