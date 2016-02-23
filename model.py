import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

import numpy as np


class WindowCell():
    def __init__(self, input_size, con_size, cluster_size, vocab_size):
        self._input_size = input_size
        self._con_size = con_size
        self._cluster_size = cluster_size
        self._vocab_size = vocab_size

    @property
    def input_size(self):
        return self._input_size #TODO

    @property
    def con_size(self):
        return self._con_size #TODO

    @property
    def state_size(self):
        return self._cluster_size #TODO

    @property
    def output_size(self):
        return self._vocab_size

    # inputs - tensor of size [batch_size x vocab_size]
    # con - tensor of size [con_size x vocab_size]
    def __call__(self, inputs, state, con, globalscope):
        with tf.variable_scope(type(self).__name__):
            # batch_size x 3(cluster_size)
            concat = linear(inputs, 3 * self._cluster_size, True)
            a, b, k = array_ops.split(1, 3, concat)
            ao = tf.exp(a)
            bo = tf.exp(b)
            ko = state + tf.exp(k)

            phi = []
            for i in range(self._con_size):
                # each phi is [batch_size x 1]
                phi.append(tf.reduce_sum(ao * tf.exp(- bo * tf.square(
                    ko - tf.constant(i, dtype=tf.float32, shape=tf.shape(k)))), 1))

            # [batch_size x vocab_size]
            wt = tf.matmul(tf.concat(1, phi), con)
        return wt, k0


class Network():
    def __init__(self, cell_fn, input_size, con, hidden_unit):
        self._con = con
        self._con_size = tf.shape(con)[1]

        self.h1 = cell_fn(hidden_unit , 1.0, input_size + self._con_size)
        self.w = WindowCell() #TODO

        self.hn = []
        for i in range(2):
            self.hn.append(cell_fn(hidden_unit , 1.0, input_size + hidden_unit + self.w.output_size))

        pass

    def __call__(self, inputs, state, scope=None):
        cur_state_pos = 0
        new_states = []

        # state
        # h1, w, h2, h3
        # 2 x hidden_unit, cluster_size, 2 x hidden_unit, 2 x hidden_unit
        with vs.variable_scope(scope or type(self).__name__):  
            #wt_1 = ??array_ops.slice(state, [0, h1.state_size], [-1, w.state_size])

            with tf.variable_scope("hidden0"):
                cur_state = array_ops.slice(state, [0, cur_state_pos], [-1, h1.state_size])
                cur_inp, new_state = self.h1(array_ops.concat(1, [inputs, wt_1]), cur_state)

                new_states.append(new_state)
                cur_state_pos = += h1.state_size

            with tf.variable_scope("window"):
                cur_state = array_ops.slice(state, [0, cur_state_pos], [-1, w.state_size])
                wt, new_state = self.w(array_ops.concat(1, [cur_inp, con]), cur_state)

                new_states.append(new_state)
                cur_state_pos = += w.state_size

            for i in range(len(self.hn)):
                with tf.variable_scope("hidden%d" % (i+1)):
                    cur_state = array_ops.slice(state, [0, cur_state_pos], [-1, self.hn[i].state_size])
                    cur_inp, new_state = self.hn[i](array_ops.concat(1, [inputs, cur_inp, wt]), cur_state)
                    new_states.append(new_state)
                    cur_state_pos = += self.hn[i].state_size


        return cur_inp, array_ops.concat(1, new_states)


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

        cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
          
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
            #embedding = tf.constant(np.identity(args.vocab_size, dtype=np.float32))

            # embedding_lookup is like slicing matrix by index
            # split along dimension 1 (0-index) with #split = seq_length
            inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))

            # remove dimensions with dimension 1 from the shape of tensor at the second dimension(the [1] parameter).
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]


        def loop(prev, _):
            prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, states = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')

        # turn a list of output into row matrix where each row is output
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])

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

    def sample(self, sess, chars, vocab, num=200, prime='The '):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in xrange(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]
            # sample = int(np.random.choice(len(p), p=p))
            sample = weighted_pick(p)
            pred = chars[sample]
            ret += pred
            char = pred
        return ret


class Model():
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

        cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            with tf.device("/cpu:0"):
                #embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                embedding = tf.constant(np.identity(args.vocab_size, dtype=np.float32))

                # embedding_lookup is like slicing matrix by index
                # split along dimension 1 (0-index) with #split = seq_length
                inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))

                # remove dimensions with dimension 1 from the shape of tensor at the second dimension(the [1] parameter).
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]


        def loop(prev, _):
            prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, states = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')

        # turn a list of output into row matrix where each row is output
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])

        self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        print "{}"
        print states
        print "{}"
        #self.final_state = states[-1]
        self.final_state = states
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, chars, vocab, num=200, prime='The '):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in xrange(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]
            # sample = int(np.random.choice(len(p), p=p))
            sample = weighted_pick(p)
            pred = chars[sample]
            ret += pred
            char = pred
        return ret
