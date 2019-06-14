import numpy as np
from keras.utils import np_utils



def process(filename):
    # filename = input('Enter name of file')
    assert isinstance(filename, str), 'Filename should be string'

    try:
        whole_text = open(filename).read()
    except:
        print('Error: No file found Note:enter filename with extension')

    character = sorted(list(set(whole_text)))

    idx2chr = {n: char for n, char in enumerate(character)}
    chr2idx = {char: n for n, char in enumerate(character)}

    X = []
    Y = []
    length = len(whole_text)
    seq_length = 100

    for i in range(0, length-seq_length, 1):
        sequence = whole_text[i:i + seq_length]
        label =whole_text[i + seq_length]
        X.append([chr2idx[char] for char in sequence])
        Y.append(chr2idx[label])

    X_modified = np.reshape(X, (len(X), seq_length, 1))
    X_modified = X_modified / float(len(character))
    Y_modified = np_utils.to_categorical(Y)

    return X, Y, character, idx2chr, chr2idx, X_modified, Y_modified


