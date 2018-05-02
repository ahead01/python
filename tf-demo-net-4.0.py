# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:53:12 2018

@author: Austin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  1 22:03:29 2018

@author: Austin
"""

'''
Steps
    1. input > weight > hidden layer 1 ( activation function ) > weights > hidden layer 2 (activation function) ... output layer

compare output to intended output > cost(aka loss) function (cross entropy)
optimization function - attempts to minimize the cost (AdamOptimises, SGD...)

backpropogation
feed forward + backprop = epoch


'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

# get my own data
csv_red = r'C:\Users\Austin\Towson\DataMining\Assingnments\Term_Project\DataSets\wine\orig-csv\winequality-red.csv'
csv_white=r'C:\Users\Austin\Towson\DataMining\Assingnments\Term_Project\DataSets\wine\orig-csv\winequality-white.csv'
data_source = csv_white
    
my_data = np.genfromtxt(data_source, delimiter=',', skip_header=1)
features = my_data[:,0:10]
labels= my_data[:,11]

def convert_to_one_hot(int_array):
    result = []
    for element in int_array:
        element = math.floor(element)
        row = np.zeros(10,dtype='int')
        row[element] = 1
        result.append(row)
    return np.array(result)

one_hot_labels = convert_to_one_hot(labels)   
    
#data=my_data
#test_size=.25
def my_train_test_split(data, test_size):
    train_x = np.array([])
    train_y = np.array([])
    test_x = np.array([])
    test_y = np.array([])
    rows, cols = np.shape(data)
    test_rows = math.floor(test_size * rows)
    train_rows = rows - test_rows
    np.random.shuffle(data)
    train_x, test_x = data[:train_rows,0:(cols-1)], data[train_rows:,0:(cols-1)]
    train_y, test_y = data[:train_rows,(cols-1)], data[train_rows:,(cols-1)]
    train_y = convert_to_one_hot(train_y)
    test_y = convert_to_one_hot(test_y)
    return train_x, test_x, train_y, test_y

train_x, test_x, train_y, test_y = my_train_test_split(my_data, test_size=.25)


plt.hist(labels, bins=10)
plt.title("Histogram of labels")
plt.show()


# demo data
#mnist = input_data.read_data_sets('c:/Users/Austin/code/python/datasets', one_hot=True)

# 10 classes 0-9
n_classes = 10
n_features = 11

n_nodes_in_layer = 11
n_nodes_hl1 = 200
n_nodes_hl2 = 200
n_nodes_hl3 = 200
n_nodes_hl4 = 200
n_nodes_out_layer = n_classes
initial_bias = 1

batch_size = 5

# height * width - specify input shape
x = tf.placeholder('float') # data
y = tf.placeholder('float') # labels


#number of weights= length of input (784) - all weights in a 'tensor' aka array - this is a seed basically
in_layer_dtls = {
        'weights':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_in_layer])),
        'biases':tf.Variable(tf.ones([n_nodes_in_layer]))
        }
hidden_1_layer_dtls = {
        'weights':tf.Variable(tf.random_normal([n_nodes_in_layer,n_nodes_hl1])),
        'biases':tf.Variable(tf.ones([n_nodes_hl1]))
        }
hidden_2_layer_dtls = {
        'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
        'biases':tf.Variable(tf.ones([n_nodes_hl2]))
        }
hidden_3_layer_dtls = {
        'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
        'biases':tf.Variable(tf.ones([n_nodes_hl3]))
        }
hidden_4_layer_dtls = {
        'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl4])),
        'biases':tf.Variable(tf.ones([n_nodes_hl4]))
        }
out_layer_dtls = {
        'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_out_layer])),
        'biases':tf.Variable(tf.ones([n_nodes_out_layer]))
        }



# define model of the network
def nureal_network_model(data):
    


     # (input data * weights ) + bias
    #  layer 1 = data * weights + bias with relu
#    print (data.get_shape())
#    print(type(data))
#    print(in_layer_dtls['weights'].get_shape())
#    print(type(in_layer_dtls['weights']))
    in_layer = tf.nn.relu_layer(data, in_layer_dtls['weights'], in_layer_dtls['biases'] )
#    in_layer = tf.add(tf.matmul(data, in_layer_dtls['weights']), in_layer_dtls['biases'])
#    in_layer = tf.nn.relu(in_layer)
    #  layer 2 = layer 1 * weights + bias with relu
    hl1 = tf.nn.relu_layer(in_layer, hidden_1_layer_dtls['weights'], hidden_1_layer_dtls['biases'] )
    #  layer 3 = layer 2 * weights + bias with relu
    hl2 = tf.nn.relu_layer(hl1, hidden_2_layer_dtls['weights'], hidden_2_layer_dtls['biases'] )
    hl3 = tf.nn.relu_layer(hl2, hidden_3_layer_dtls['weights'], hidden_3_layer_dtls['biases'] )
    hl4 = tf.nn.relu_layer(hl3, hidden_4_layer_dtls['weights'], hidden_4_layer_dtls['biases'] )
    #output layer = l3 * weight + bias
    out_layer = tf.matmul(hl4, out_layer_dtls['weights']) + out_layer_dtls['biases']
    
    return out_layer

def train_nn(x):
    
    prediction = nureal_network_model(x)
    # 
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    #optimizer
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(.0001).minimize(cost)

    hm_epochs = 10
    
    #train run
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
        print(sess.run(c))
        # OLD
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            #for _ in range(int(mnist.train.num_examples/batch_size)):
                # features and labels defined here
                #features_x, labels_y = mnist.train.next_batch(batch_size)
            i=0
            while i < np.shape(train_x)[0]:
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
#                print(type(batch_x))
#                print(batch_x)
                _, c = sess.run([optimizer, cost], feed_dict = { x: batch_x , y: batch_y })
                epoch_loss += c
                i += batch_size
            #accuracy = tf.metrics.accuracy()
            print('Epoch', epoch, 'completed out of ', hm_epochs, 'loss' , epoch_loss)
            #tf.argmax returns index of max value in array
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            # 
            print('Accuracy', accuracy.eval({x:test_x, y:test_y }))
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # 
        print('Accuracy', accuracy.eval({x:test_x, y:test_y }))
      
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
train_nn(x)
    






