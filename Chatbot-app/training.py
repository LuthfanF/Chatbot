import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Flatten
from keras.optimizers import SGD
import random

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('data.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize and lowercase each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))

pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# create our training data
training = []
output_empty = [0] * len(classes)

# create an empty array for our output
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# convert train_x and train_y to numpy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

# define input shape
input_shape = len(train_x[0])
# define vocabulary size
vocabulary = len(words)
# define output length
output_length = len(train_y[0])

# create the model
model = Sequential()
model.add(Embedding(vocabulary, 10, input_length=input_shape))
model.add(LSTM(10, return_sequences=True))
model.add(Flatten())
model.add(Dense(output_length, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# save the model
model.save('model.h5')

print("Model created and saved.")