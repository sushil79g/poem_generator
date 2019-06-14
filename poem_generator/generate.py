from pprint import pprint
from model import Model
form pre-process import process

class Generating:
    def __init__(self,epochs=50, batch_size=64):
        self.epochs = epochs
        self.batch_size = batch_size

    def generate(self, number,save_weight=False, pretrained_weight=None,  filename = 'chanda.txt'):
        model  = Model(self.epochs, self.batch_size)
        generate_model = model.forward(filename, save_weight=False, pretrained_weight=None)
        
        X, Y, characters, idx2chr, chr2idx, X_modified, Y_modified = process(filename)
        string_mapped = X[number]
        full_string = [idx2chr[value] for value in string_mapped]
        for i in range(400):
            x = np.reshape(string_mapped,(1,len(string_mapped), 1))
            x = x / float(len(characters))
            pred_index = np.argmax(generate_model.predict(x, verbose=0))
            seq = [idx2chr[value] for value in string_mapped]
            full_string.append(idx2chr[pred_index])
            string_mapped.append(pred_index)
            string_mapped = string_mapped[1:len(string_mapped)]
        poem = ""
        for char in full_string:
            poem = poem + char
        return poem