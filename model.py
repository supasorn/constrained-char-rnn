import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

import numpy as np


class WindowCell():
    def __init__(self, input_size, vocab_size, con_size):
        self._input_size = input_size
        self._vocab_size = vocab_size
        self._con_size = con_size

    @property
    def input_size(self):
        return self._input_size 

    @property
    def con_size(self):
        return self._con_size 

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self._vocab_size

    # inputs - tensor of size [batch_size x vocab_size]
    # con - tensor of size [batch_size x seq_length x vocab_size]
    def __call__(self, inputs, state, con, timestep):
        with tf.variable_scope(type(self).__name__):

            concat = rnn_cell.linear(inputs, 1, True)
            #b, k = tf.split(1, 2, concat)
            #bo = tf.exp(b)
            #ko = state + tf.exp(k) 
            ko = state + tf.exp(concat)

            phi = []
            for i in range(self._con_size):
                # each phi is [batch_size x 1]
                #phi.append(tf.exp(- bo * tf.square(ko - i)))
                phi.append(tf.exp(- tf.square(i - ko)))


            # tf.concat(1, phi) -> [batch_size x seq_length]
            # tf.expan_dims(%, 1) -> [batch_size x 1 x seq_length]
            # tf.batch_matmul(%, con) -> [batch_size x 1 x vocab_size]
            # tf.squeeze(%) -> [batch_size x vocab_size]
            wt = tf.squeeze(tf.batch_matmul(tf.expand_dims(tf.concat(1, phi), 1), con), [1])
        return wt, ko


class Network():
    def __init__(self, cell_fn, input_size, vocab_size, hidden_unit, con_size, num_layers):
        self._num_layers = num_layers
        self._con_size = con_size
        self._vocab_size = vocab_size
        self._hidden_unit = hidden_unit
        self.w = WindowCell(input_size, vocab_size, con_size) 


        self.hn = []
        self.hn.append(cell_fn(hidden_unit , 1.0, input_size + self.w.output_size))
        for i in range(num_layers - 1):
            self.hn.append(cell_fn(hidden_unit , 1.0, input_size + self.hn[i].output_size + self.w.output_size))

        # +1 for time variable
        self._state_size = self._vocab_size + self.w.state_size + sum(h.state_size for h in self.hn) + 1

    def zero_state(self, batch_size, dtype, con):
        zeros = tf.zeros([batch_size, self._state_size - self._vocab_size - 1], dtype=dtype)

        wpart, _ = self.w(
                tf.zeros([batch_size, self._vocab_size], dtype=tf.float32), 
                tf.zeros([batch_size, self.w.state_size], dtype=tf.float32), con, 
                tf.zeros([batch_size, 1], dtype=tf.float32))

        timepart = tf.ones([batch_size, 1])
        return tf.concat(1, [wpart, zeros, timepart])

    def zero_constrain(self, batch_size):
        zeros = tf.zeros([batch_size, self._con_size], dtype=tf.float32)
        return zeros
    
    # con: [batch_size x seq_length x vocab_size]

    def __call__(self, inputs, state, con, scope=None):
        cur_state_pos = 0
        new_states = []

        # state
        # wt, h1, w, h2, h3
        # vocab_size, 2 x hidden_unit, cluster_size, 2 x hidden_unit, 2 x hidden_unit
        with tf.variable_scope(scope or type(self).__name__):  

            outh = [None] * len(self.hn)

            wt_1 = tf.slice(state, [0, 0], [-1, self._vocab_size])
            new_states.append(None) 
            cur_state_pos += self._vocab_size

            with tf.variable_scope("hidden1"):
                cur_state = tf.slice(state, [0, cur_state_pos], [-1, self.hn[0].state_size])
                outh[0], new_state = self.hn[0](tf.concat(1, [inputs, wt_1]), cur_state)

                new_states.append(new_state)
                cur_state_pos += self.hn[0].state_size

            with tf.variable_scope("window"):
                cur_state = tf.slice(state, [0, cur_state_pos], [-1, self.w.state_size])
                timestep = tf.slice(state, [0, self._state_size-1], [-1, 1])

                wt, new_state = self.w(outh[0], cur_state, con, timestep)

                new_states[0] = wt
                new_states.append(new_state)
                cur_state_pos += self.w.state_size

            for i in range(1, len(self.hn)):
                with tf.variable_scope("hidden%d" % (i+1)):
                    cur_state = tf.slice(state, [0, cur_state_pos], [-1, self.hn[i].state_size])
                    outh[i], new_state = self.hn[i](tf.concat(1, [inputs, outh[i-1], wt]), cur_state)

                    new_states.append(new_state)
                    cur_state_pos += self.hn[i].state_size

            new_states.append(timestep + 1)


        return tf.concat(1, outh), tf.concat(1, new_states) 


def decoder(inputs, initial_state, network, con, loop_function=None, scope=None):
    with tf.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(inputs):
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = network(inp, state, con)
            #output, state = network(inp, initial_state, con)
            outputs.append(output)
            if loop_function is not None:
                prev = output
    return outputs, state

class ConstrainedModel():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        #cell = cell_fn(args.rnn_size)
        con_size = 50 #args.seq_length

        #self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)
        self.network = Network(cell_fn, args.vocab_size, args.vocab_size, args.rnn_size, con_size, args.num_layers)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.con_data = tf.placeholder(tf.int32, [args.batch_size, con_size])

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size * args.num_layers, args.vocab_size])
          
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            #embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
            embedding = tf.constant(np.identity(args.vocab_size, dtype=np.float32))

            inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

            # [(batch_size * seq_length) x vocab_size]
            con = tf.nn.embedding_lookup(embedding, self.con_data)

        self.initial_state = self.network.zero_state(args.batch_size, tf.float32, con)

        def loop(prev, _):
            prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        #outputs, states = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        outputs, states = decoder(inputs, self.initial_state, self.network, con, loop_function=loop if infer else None, scope='rnnlm')

        # turn a list of output into row matrix where each row is output
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size * args.num_layers])

        self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        print states
        self.final_state = states

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, chars, vocab, num=200, prime=''):
        #con = self.network.zero_constrain(1).eval()
        con_text = prime + 'what the hell obama american'
        #con_text = prime + '' 
        #con_text = con_text.upper() + ' ' * (50 - len(con_text))
        con_text = con_text + ' ' * (50 - len(con_text))
        con = np.expand_dims(map(vocab.get, con_text), 0)
        print con
        print con_text
        init = 0
        
        state = self.initial_state.eval({self.con_data: con})
        np.set_printoptions(threshold='nan')
        #print state[0][:68]

        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]

            feed = {self.input_data: x, self.initial_state:state, self.con_data: con}

            [state] = sess.run([self.final_state], feed)
            #print state[0][:68]
            #print "max = "
            #print np.argmax(state[0][:self.network._vocab_size])
            #print state[0][668:668+20]
            pos = self.network._vocab_size + self.network.hn[0].state_size
            print state[0][pos:pos+1]


        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in xrange(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state, self.con_data: con}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            #print state[0][:68]
            p = probs[0]
            # sample = int(np.random.choice(len(p), p=p))
            pos = self.network._vocab_size + self.network.hn[0].state_size
            print state[0][pos:pos+1]

            sample = weighted_pick(p)
            pred = chars[sample]
            ret += pred
            char = pred
        return ret


