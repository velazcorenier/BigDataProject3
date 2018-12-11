import json
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
from argparse import ArgumentParser
import os

cwd = os.getcwd()

# this utility makes sure that all the words in your input
# are registered in the dictionary
# before trying to turn them into a matrix.
def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            print("'%s' not in training corpus; ignoring." %(word))
    return wordIndices

if __name__ == '__main__':
    parser = ArgumentParser(description = "CNN and Logistic Regresion for Tweet Analysis.")
    parser.add_argument('--model', default = 'TWEETNET')
    parser.add_argument('--max_words', default = 1000)
    
    args = parser.parse_args()

    # we're still going to use a Tokenizer here, but we don't need to fit it
    tokenizer = Tokenizer(num_words=1000)
    # for human-friendly printing
    labels = ['No', 'Yes', 'Ambiguous']
     
    
    # read in our saved dictionary
    with open('dictionary.json', 'r') as dictionary_file:
        dictionary = json.load(dictionary_file)    

    #===================================================================
    #=========================== LOAD MODEL ============================
    #===================================================================
    
    # read in your saved model structure
    json_file = open(cwd + '/models/model_' + args.model + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # and create a model from that
    model = model_from_json(loaded_model_json)
    # and weight your nodes with your saved values
    model.load_weights(cwd + '/models/model_' + args.model + '.h5')

    #====================================================================
    #============================ LOAD DATA =============================
    #====================================================================

    # Extract data from a csv
    data = np.genfromtxt(cwd + '/data/tweetTexts.csv', delimiter=',', usecols=(0), dtype=None)
    inputs = [x[0] for x in data]

    allWordIndices = []
    #tokenizer.fit_on_texts(inputs)

    # for each tweet, change each token to its ID in the Tokenizer's word_index
    for text in data:
        wordIndices = convert_text_to_index_array(text)
        allWordIndices.append(wordIndices)
    
    allWordIndices = np.asarray(allWordIndices)
    inputs = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
    
    results = model.predict(inputs)
    
    results = np.apply_along_axis(np.argmax, axis=1, arr=results)
    results = np.expand_dims(results, axis=1)
    
    data = np.expand_dims(data, axis=1)

    results = np.hstack((data, results))

    np.savetxt(cwd + "/results/results_" + args.model + ".csv", results, delimiter=",", fmt='%s')
