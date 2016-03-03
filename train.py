import numpy as np
import tensorflow as tf

import argparse
import time
import os
import cPickle

from utils import TextLoader
from model import ConstrainedModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=300,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=5,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--reprocess', type=int, default=0,
                       help='reprocess input')
    args = parser.parse_args()
    #train(args)
    train2(args)

def decayForEpoch(args, e):
    return args.learning_rate * (args.decay_rate ** e)

def train2(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length, args.reprocess)
    args.vocab_size = data_loader.vocab_size

    totalTask = args.num_epochs * data_loader.num_batches

    lastCheckpoint = tf.train.latest_checkpoint(args.save_dir) 
    if lastCheckpoint is None:
        startEpoch = 0
    else:
        print "Last checkpoint :", lastCheckpoint
        startEpoch = int(lastCheckpoint.split("-")[-1])

    print "startEpoch = ", startEpoch

    with open(os.path.join(args.save_dir, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'w') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    model = ConstrainedModel(args)

    etaCount = 0
    etaString = "-" 
    etaStart = time.time()
    etaTime = 0
    duration = 0

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())



        if startEpoch > 0: # load latest checkpoint
            print "Loading last checkpoint"
            saver.restore(sess, lastCheckpoint)

        for e in xrange(startEpoch, args.num_epochs):
            sess.run(tf.assign(model.lr, decayForEpoch(args, e)))
            data_loader.reset_batch_pointer()

            for b in xrange(data_loader.num_batches):
                start = time.time()
                x, y, con = data_loader.next_batch()
                #print x, y, con
                #exit(0)

                state = model.initial_state.eval({model.con_data: con})
                #print state[:, :68]
                #print con
                #np.savetxt("state.txt", state[:, :68], "%.2f")
                #np.savetxt("con.txt", con, "%.2f")
                #exit(0)

                feed = {model.input_data: x, model.targets: y, model.initial_state: state, model.con_data:con}

                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                #time.sleep(0.01)
                #train_loss = 5
                end = time.time()

                taskNum = (e * data_loader.num_batches + b)
                etaCount += 1
                if (etaCount) % 25 == 0:
                    duration = time.time() - etaStart
                    etaStart = time.time()

                etaTime = (totalTask - (taskNum + 1)) / 25 * duration
                m, s = divmod(etaTime, 60)
                h, m = divmod(m, 60)
                etaString = "%d:%02d:%02d" % (h, m, s)

                print "{}/{} (epoch {}), loss = {:.3f}, time/batch = {:.3f}, ETA: {} ({})" \
                    .format(taskNum, totalTask, e, train_loss, end - start, time.strftime("%H:%M:%S", time.localtime(time.time() + etaTime)), etaString)

            if (e + 1) % args.save_every == 0 or e == args.num_epochs - 1:
                checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = e + 1)
                print "model saved to {}".format(checkpoint_path)


if __name__ == '__main__':
    main()
