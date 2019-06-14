from keras.layers import RNN
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import Sequential

from pre-process import process

class Model:
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size

    def build(self, X_modified1, X_modified2, Y_modified):
        model = Sequential()
        model.add(LSTM(400, input_shape=(X_modified1, X_modified2), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(400, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(400))
        model.add(Dropout(0.2))
        model.add(Dense(Y_modified, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        return model

    def forward(self, filename, save_weight=False, pretrained_weight=None):
        X, Y, character, idx2chr, chr2idx, X_modified, Y_modified = process(filename)
        model = self.build(X_modified.shape[1], X_modified.shape[2],Y_modified.shape[1])

        if pretrained_weight is not None:
            model.load_weights(pretrained_weight)
        else:
            model.fit(X_modified, Y_modified, epochs=self.epochs, batch_size=self.batch_size)
        
        if save_weight:
            model.sample_weights('poem_generator_weight.h5')
        
        return model