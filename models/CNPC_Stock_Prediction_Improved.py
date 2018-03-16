#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# hyper parameters
input_neurons=11
hidden_neurons=6
output_neurons=1
learning_rate=0.0003
split_node=2300
sequence_len_in=10
sequence_len_out=10
batch_size=100
epoch=120

# import data
data = pd.read_csv('CNPC.csv').iloc[:, 1:13].values  # columns 2-12

# RNN parameters
weights = {'in': tf.Variable(tf.random_normal([input_neurons, hidden_neurons])),
            'out': tf.Variable(tf.random_normal([hidden_neurons, output_neurons]))}
biases = {'in': tf.Variable(tf.constant(0.01, shape=[hidden_neurons, ])),
            'out': tf.Variable(tf.constant(0.01, shape=[output_neurons, ]))}

# normalization
def normalize_col(data):
    return np.mean(data, axis=0), np.std(data, axis=0), (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def normalize_row(data):
    return (data - np.mean(data, axis=1)[:, np.newaxis]) / np.std(data, axis=1)[:, np.newaxis]

# fetch training data
def fetch_training_data(data=data, sequence_len_in=sequence_len_in, sequence_len_out=sequence_len_out, batch_size=batch_size, split_node=split_node):
    batch_idx=[]
    training_feat=[]
    training_tar=[]
    training_data=data[:split_node]
    _, _, norm_training_data=normalize_col(training_data)
    rows=len(norm_training_data)
    for i in range(rows-sequence_len_in-sequence_len_out):
        X=norm_training_data[i:i+sequence_len_in, :-1]
        X=normalize_row(X)
        Y=norm_training_data[i+sequence_len_in:i+sequence_len_in+sequence_len_out, -1, np.newaxis]
        training_feat.append(X.tolist())
        training_tar.append(Y.tolist())
        if i % batch_size==0:
            batch_idx.append(i)
    batch_idx.append((rows-sequence_len_in-sequence_len_out))
    return training_feat, training_tar, batch_idx

def fetch_test_data(data=data, sequence_len_in=sequence_len_in, sequence_len_out=sequence_len_out, split_node=split_node):
    test_feat=[]
    test_tar=[]
    test_data=data[split_node:]
    mean, std, norm_test_data=normalize_col(test_data)
    rows = len(norm_test_data)
    for i in range(rows-sequence_len_in-sequence_len_out):
        X=norm_test_data[i:i+sequence_len_in, :-1]
        X=normalize_row(X)
        Y=norm_test_data[i+sequence_len_in:i+sequence_len_in+sequence_len_out, -1, np.newaxis]
        test_feat.append(X.tolist())
        test_tar.append(Y)
    return test_feat, test_tar, mean, std

def RNN_model(features):   # features: batch_size x sequence_len x input_neuron
    matrix=tf.reshape(features, [-1, input_neurons])
    matrix=tf.matmul(matrix, weights['in'])+biases['in']
    input_matrix=tf.reshape(matrix, [-1, tf.shape(features)[1], hidden_neurons])
    rnn_cell=tf.nn.rnn_cell.BasicLSTMCell(hidden_neurons)
    initial_state=rnn_cell.zero_state(tf.shape(features)[0], dtype=tf.float32)
    output, final_states = tf.nn.dynamic_rnn(rnn_cell, input_matrix, initial_state=initial_state, dtype=tf.float32)
    output=tf.reshape(output, [-1, hidden_neurons])
    prediction=tf.matmul(output, weights['out'])+biases['out']
    prediction=tf.reshape(prediction, [-1, tf.shape(features)[1], output_neurons])
    return prediction

def training():
    X = tf.placeholder(tf.float32, shape=[None, sequence_len_in, input_neurons])
    Y = tf.placeholder(tf.float32, shape=[None, sequence_len_out, output_neurons])
    training_feat, training_tar, batch_idx = fetch_training_data()
    with tf.variable_scope("model"):
        prediction = RNN_model(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(prediction, [-1, output_neurons]) - tf.reshape(Y, [-1, output_neurons])))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            for step in range(len(batch_idx)-1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: training_feat[batch_idx[step]:batch_idx[step + 1]],
                                                                 Y: training_tar[batch_idx[step]:batch_idx[step + 1]]})
            print("Number of iterations:", i, " loss:", loss_)
        print("model_save: ", saver.save(sess, 'RNN_model_save\\model.ckpt'))

def predicting():
    X = tf.placeholder(tf.float32, shape=[None, sequence_len_in, input_neurons])
    test_feat, test_tar, mean, std = fetch_test_data()
    with tf.variable_scope("model", reuse=True):
        prediction = RNN_model(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint('RNN_model_save')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_feat)-1):
            pred = sess.run(prediction, feed_dict={X: [test_feat[step]]})
            pred=np.array(pred).reshape((-1, output_neurons))
            test_predict.append(pred)
        test_tar = np.array(test_tar) * std[-1] + mean[-1]
        test_tar=test_tar[1:]
        test_predict = np.array(test_predict) * std[-1] + mean[-1]
        print(test_tar)
        print(test_predict)
        #print(len(test_predict))
        #print(len(test_tar))
        accuracy = np.average(np.abs(test_predict - test_tar), axis=1)
        print((accuracy))
    return test_predict, test_tar

training()
pre, tar=predicting()

plt.plot(tar[0], color='r', label='Real')
plt.plot(pre[0], color='g', label='Predicted')
plt.legend(loc='upper left')
plt.show()