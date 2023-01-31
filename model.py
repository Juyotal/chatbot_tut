import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import nltk
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer


stemmer = PorterStemmer()
ignore_words = ['?', '.', '!']

def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

class ChatDataset(Dataset):

    def __init__(self, path):
        self.words = []
        self.labels = []
        self.xy = []
        self.x_data = []
        self.y_data = []

        with open(path, 'r') as f:
            intents = json.load(f)

        for intent in intents['intents']:
            tag = intent['tag']
            # add to tag list
            self.labels.append(tag)

            for pattern in intent['patterns']:
                # tokenize each word in the sentence
                w = nltk.word_tokenize(pattern)
                # add to our words list
                self.words.extend(w)
                # add to xy pair
                self.xy.append((w, tag))

        self.words = [stemmer.stem(w.lower()) for w in self.words if w not in ignore_words]

        # remove duplicates and sort
        self.words = sorted(set(self.words))
        self.labels = sorted(set(self.labels))

        for (pattern_sentence, tag) in self.xy:
            # X: bag of words for each pattern_sentence
            bag = bag_of_words(pattern_sentence, self.words)
            self.x_data.append(bag)
            # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
            label = self.labels.index(tag)
            self.y_data.append(label)

        self.n_samples = len(self.x_data)

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

