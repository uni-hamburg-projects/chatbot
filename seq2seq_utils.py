from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop
#from nltk import FreqDist
import numpy as np
import os
import datetime
import re

def load_data(train_source, train_dist, test_source, test_dist, max_len, vocab_size):
    '''
    fin = open(test_source, "r")
    data2 = fin.read()
    fin.close()
    fout = open(train_source, "a")
    fout.write(data2)
    fout.close()

    fin = open(test_dist, "r")
    data2 = fin.read()
    fin.close()
    fout = open(train_dist, "a")
    fout.write(data2)
    fout.close()
    '''
    
    # Reading raw text from source and destination files
    f = open(train_source, 'r')
    X_data = f.read()
    f.close()
    f = open(train_dist, 'r')
    y_data = f.read()
    f.close()

    # Splitting raw text into array of sequences
    X = [text_to_word_sequence(x)[::-1] for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]
    y = [text_to_word_sequence(y) for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]

    #Check or Create Vocab 
    vocab_files = [f for f in os.listdir('.') if 'vocab' in f]
    x_vocab_file = open(os.path.join('', 'vocab_x.txt'), 'a+')
    y_vocab_file = open(os.path.join('', 'vocab_y.txt'), 'a+')
    if len(vocab_files) == 0:    
        vocab_x = {}
        for line in X:
            for token in line:
                if not token in vocab_x:
                    vocab_x[token] = 0
                vocab_x[token] += 1

        X_vocab = sorted(vocab_x, key=vocab_x.get, reverse=True)
        X_vocab = X_vocab[0:(vocab_size)]
        for (i, item) in enumerate(X_vocab):
            if item == "newlinechar":
                X_vocab[i] = "-"
        for item in X_vocab:
            print>>x_vocab_file, item
        x_vocab_file.close()

        vocab_y = {}
        for line in y:
            for token in line:
                if not token in vocab_y:
                    vocab_y[token] = 0
                vocab_y[token] += 1

        y_vocab = sorted(vocab_y, key=vocab_y.get, reverse=True)
        y_vocab = y_vocab[0:(vocab_size)]
        for (i, item) in enumerate(y_vocab):
            if item == "newlinechar":
                y_vocab[i] = "-"
        for item in y_vocab:
            print>>y_vocab_file, item
        y_vocab_file.close()
    else:
        X_vocab = x_vocab_file.read().splitlines()
        y_vocab = y_vocab_file.read().splitlines()
        
    # Creating the vocabulary set with the most common words
    #dist = FreqDist(np.hstack(X))
    #X_vocab = dist.most_common(vocab_size-1)
    #dist = FreqDist(np.hstack(y))
    #y_vocab = dist.most_common(vocab_size-1)

    # Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
    X_ix_to_word = X_vocab
    # Adding the word "ZERO" to the beginning of the array
    X_ix_to_word.insert(0, 'ZERO')
    # Adding the word 'UNK' to the end of the array (stands for UNKNOWN words)
    X_ix_to_word.append('UNK')

    # Creating the word-to-index dictionary from the array created above
    #X_word_to_ix = {word:ix for ix, word in enumerate(X_ix_to_word)}
   
    X_word_to_ix = dict((map(reversed, enumerate(X_ix_to_word))))

    # Converting each word to its index value
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                X[i][j] = X_word_to_ix[word]
            else:
                X[i][j] = X_word_to_ix['UNK']

    y_ix_to_word = y_vocab
    y_ix_to_word.insert(0, 'ZERO')
    y_ix_to_word.append('UNK')
    
    #y_word_to_ix = {word:ix for ix, word in enumerate(y_ix_to_word)}
    
    y_word_to_ix = dict((map(reversed, enumerate(y_ix_to_word))))
    
    for i, sentence in enumerate(y):
        for j, word in enumerate(sentence):
            if word in y_word_to_ix:
                y[i][j] = y_word_to_ix[word]
            else:
                y[i][j] = y_word_to_ix['UNK']
    
    return (X, len(X_vocab), X_word_to_ix, X_ix_to_word, y, len(y_vocab), y_word_to_ix, y_ix_to_word)

def load_test_data(source, X_word_to_ix, max_len):
    f = open(source, 'r')
    X_data = f.read()
    f.close()

    X = [text_to_word_sequence(x)[::-1] for x in X_data.split('\n') if len(x) > 0 and len(x) <= max_len]
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                X[i][j] = X_word_to_ix[word]
            else:
                X[i][j] = X_word_to_ix['UNK']
    return X

def create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, hidden_size, num_layers):
    model = Sequential()

    # Creating encoder network
    model.add(Embedding(X_vocab_len, 1000, input_length=X_max_len, mask_zero=True))
    model.add(LSTM(hidden_size))
    model.add(RepeatVector(y_max_len))

    # Creating decoder network
    for _ in range(num_layers):
        model.add(LSTM(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(y_vocab_len)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
    return model

def process_data(word_sentences, max_len, word_to_ix):
    # Vectorizing each element in each sequence
    sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
    for i, sentence in enumerate(word_sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1.
    return sequences

def find_checkpoint_file(folder):
    checkpoint_file = [f for f in os.listdir(folder) if 'checkpoint' in f]
    if len(checkpoint_file) == 0:
        return []
    modified_time = [os.path.getmtime(f) for f in checkpoint_file]
    return checkpoint_file[np.argmax(modified_time)]

