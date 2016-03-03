import os
import collections
import cPickle
import numpy as np

# tensor contains [id0, id1, ...] where id corresponds to id of vocab
# in this case, vocab is character i.e. a, b, c, A, B, C

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, reprocess):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")
        con_file = os.path.join(data_dir, "con.npy")

        if reprocess or not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print "reading text file"
            self.preprocess(input_file, vocab_file, tensor_file, con_file)
        else:
            print "loading preprocessed files"
            self.load_preprocessed(vocab_file, tensor_file, con_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file, con_file):
        with open(input_file, "r") as f:
            data = f.read()

        # counter = { 'a': 2, 'b': 3, ...}
        counter = collections.Counter(data)
        
        # sort by frequency
        # count_pairs = [('b', 3), ('a', 2), ...]
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])

        # unzip 
        self.chars = [x[0] for x in count_pairs]

        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        #con = data.upper()
        con = data

        with open(vocab_file, 'w') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(map(self.vocab.get, data))
        self.con = np.array(map(self.vocab.get, con))

        np.save(tensor_file, self.tensor)
        np.save(con_file, self.con)

    def load_preprocessed(self, vocab_file, tensor_file, con_file):
        with open(vocab_file) as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.con = np.load(con_file)
        self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)

    def create_batches(self):
        self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)
        # truncate data so it's divisible
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        self.con = self.con[:self.num_batches * self.batch_size * self.seq_length]

        xdata = self.tensor
        ydata = np.copy(self.tensor)

        # so that, y_i is the next character of x_i
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]

        #    num_batches = 3
        #    seq_length = 4
        #    batch_size = 2

        #   AAAA|BBBB|CCCC
        #   aaaa|bbbb|cccc

        # first batch is AAAA, aaaa
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.con_batches = np.split(self.con.reshape(self.batch_size, -1), self.num_batches, 1) 


    def next_batch(self):
        x, y, con = self.x_batches[self.pointer], self.y_batches[self.pointer], self.con_batches[self.pointer]

        self.pointer += 1
        return x, y, con

    def reset_batch_pointer(self):
        self.pointer = 0

