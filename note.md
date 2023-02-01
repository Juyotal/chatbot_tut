# Code Report

## Introduction.
The Project in this Repo is a NLP representation of a chatbot meant to answer science questions.

## Requirements.
the main required Libraries in the building of our project are:
+ numpy
+ pytorch
+ nltk
+ requests 
+ bs4

## Implementation stages.
there are different stages to be executed in the implementation of such a chatbot.

1 . Creating our Dataset together with the intents (/intents.json).

2 . We extract our dataSet and PreProccess it for training by          Tokenizing, stemming and organising into Bag of Words. (/model.py)

3 . We define our Neural Network Model and fineTune it on our sentence-intent dataset. (/model.py)

4 . We train our Model with data from our intents.json when give in a sentence, expect it to determine a tag based on sequence in BOW and rectify it continously to reduce error. (/train.py)

5 . Once you have a model that performs well on the test set, it is be used to answer new questions. The model can answer new questions by inputting the question into the model and then extracting the predicted answer from the output. (/chat.py)

## Limitation of our Model.
Okay! So, the best output we can get from our Model when a text is input is find out which intent the sentence falls under. when the intent is greating or bye, we just pick up a random reply from pre determined corresponding replies. 
But in the case it is actually a question, unfortunately, we do not have a enough sample data to build a neural network which will provide an answer to most of the questions. Also, building such a neural network will either have to make use of some Pre-Trained Model such as BERT or be very time-Consuming and Computationally expensive. 


## Scraping for answer to our question.

What I chose to do as an alternative is, once the text is input and detected to be of "question" intent, a search query is performed on google and the best available option is extracted and returned as our AI reply to the question. (/scrape_for_answer.py)


## Conclusion. 

Though from our Model we are able to detect the intent to corresponding text with pretty good accuracy, our Model performs better in detecting questions intents with questions of the type *"WHAT IS ....?"* else, it is has some difficulties classifying it under a question tag. And Odds are, if our questions are classified under question intent, Scraping their answer from google search will return a reasonable response.
Also, As it is, unfortunately, our Model wont answer to only Science to questions but to a larger array as long as it is able to detect is a text of class *question* intent. this Happens as the Model does not have enough dataset and havent been trained enough to detect what specifically are science questions with more precision.

It definitely is not convenient having our answers continously scraped from the internet.  A better alternative would be having different intent classes for different Science Field topics. For greater Precision. 