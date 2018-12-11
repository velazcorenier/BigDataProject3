import os
from h5py import File
from tqdm import tqdm
from models import *
from keras.utils import to_categorical as one_hot
from argparse import ArgumentParser
from time import clock
from keras.utils import plot_model

import numpy as np
import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from argparse import ArgumentParser

import os
cwd = os.getcwd()

if __name__ == '__main__':
    parser = ArgumentParser(description = "Logistic Regression Tweet classification.")
    parser.add_argument('--model', default = 'TWEETNET')

    args = parser.parse_args()
    #=================================================================
    #======================== PREPARING DATA =========================
    #=================================================================

    # Extract data from a csv
    training = np.genfromtxt(cwd + '/data/training/cleantextlabels7.csv', delimiter=',', usecols=(0, 1), dtype=None)

    # create our training data from the tweets
    train_x = [x[0] for x in training]
    # index all the classification labels
    train_y = np.asarray([x[1] for x in training])

    # only work with the 3000 most popular words found in our dataset
    max_words = 1000
    # number of classes
    num_labels = 3
    # create a new Tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    # feed our tweets to the Tokenizer
    tokenizer.fit_on_texts(train_x)

    # Tokenizers come with a convenient list of words and IDs
    dictionary = tokenizer.word_index
    # Let's save this out so we can use it later
    with open('dictionary.json', 'w') as dictionary_file:
        json.dump(dictionary, dictionary_file)

    def convert_text_to_index_array(text):
        # one really important thing that `text_to_word_sequence` does
        # is make all texts the same length -- in this case, the length
        # of the longest text in the set.
        return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

    allWordIndices = []
    # for each tweet, change each token to its ID in the Tokenizer's word_index
    for text in train_x:
        wordIndices = convert_text_to_index_array(text)
        allWordIndices.append(wordIndices)

    # now we have a list of all tweets converted to index arrays.
    # cast as an array for future usage.
    allWordIndices = np.asarray(allWordIndices)

    # create one-hot matrices out of the indexed tweets
    train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
    # treat the labels as categories
    train_y = keras.utils.to_categorical(train_y, num_labels)

    #================================================================
    #======================== TRAIN NETWORK =========================
    #================================================================
    
    if args.model == 'TWEETNET':
        model = TWEETNET(max_words, num_labels)
    
    if args.model == 'BESTNET':
        model = BESTNET(max_words, num_labels)

    model.fit(train_x, train_y,
      batch_size=32,
      epochs=10,
      verbose=1,
      validation_split=0.2,
      shuffle=True)

    #================================================================
    #======================== SAVE MODEL =========================
    #================================================================

    model_json = model.to_json()
    with open(cwd + '/models/model_' + args.model + '.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(cwd + '/models/model_' + args.model + '.h5')
