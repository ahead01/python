# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:56:42 2018

@author: Austin
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution() #- restart kernal if this gets buggy

import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe
csv_red = r'C:\Users\Austin\Towson\DataMining\Assingnments\Term_Project\DataSets\wine\orig-csv\winequality-red.csv'
csv_white=r'C:\Users\Austin\Towson\DataMining\Assingnments\Term_Project\DataSets\wine\orig-csv\winequality-white.csv'

data_source = csv_red

glbl_learning_rate=0.01 # for the input to the optimizer function
print("Eager execution: {}".format(tf.executing_eagerly()))

# csv parsing function
def parse_csv(line):
    #E7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5
  example_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0]]  # sets field types
  parsed_line = tf.decode_csv(line, example_defaults)
  print('Type :' + str(type(parsed_line)))
  print(str(parsed_line[11]))
  # First 11 fields are features, combine into single tensor parse
  features = tf.reshape(parsed_line[:-1], shape=(11,))
  # Last field is the label -# shape [] reshapes to a scalar
  label = tf.reshape(parsed_line[-1], shape=())
  return features, label



# pre-process the training dataset
train_dataset = tf.data.TextLineDataset(data_source)
train_dataset = train_dataset.skip(1)             # skip the first header row
train_dataset = train_dataset.map(parse_csv)      # parse each row
train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
train_dataset = train_dataset.batch(32)

# View a single example entry from a batch
feature, label = tfe.Iterator(train_dataset).next()
print("example features:", feature)
print("example label:", label)
print("example features:", feature[1])
print("example label:", label[1])

#The tf.keras.Sequential model is a linear stack of layers. 
    #Its constructor takes a list of layer instances.
    #Here; two Dense layers with 10 nodes each, and an output 
    #layer with 3 nodes representing our label predictions. 
    #The first layer's input_shape parameter corresponds to 
    #the amount of features from the dataset, and is required.
model = tf.keras.Sequential([
  tf.keras.layers.Dense(11, activation="relu", input_shape=(11,)),  # input shape required
  tf.keras.layers.Dense(200, activation="relu"),
  tf.keras.layers.Dense(200, activation="relu"),
  tf.keras.layers.Dense(10)
])
    
       # Define  a function of the error - takes in the prediction and true label 
def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

    # Define the gradient function - this guy calls the loss function
def grad(model, inputs, targets):
  with tfe.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

# Gradient dedcent style optimizer
  
optimizer = tf.train.GradientDescentOptimizer(learning_rate=glbl_learning_rate)

#A training loop feeds the dataset examples into the model to help it make better predictions. The following code block sets up these training steps:
#
# 1. Iterate each epoch. An epoch is one pass through the dataset.
# 2. Within an epoch, iterate over each example in the training Dataset grabbing its features (x) and label (y).
# 3. Using the example's features, make a prediction and compare it with the label.
#       Then measure the inaccuracy of the prediction and use that to calculate the model's loss and gradients.
# 4. Use an optimizer to update the model's variables.
# 5. Keep track of some stats for visualization.
# 6. Repeat for each epoch.

## Note: Rerunning this cell uses the same model variables

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

# aka. loops
num_epochs = 500

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

  # Training loop - using batches of 32
    for x, y in tfe.Iterator(train_dataset):
        #print('x:'+ str(x[0]) + ' y: ' + str(y[0]))
        # Optimize the model 
        grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),
                              global_step=tf.train.get_or_create_global_step())
    
        # Track progress
        epoch_loss_avg(loss(model, x, y))  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    #if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                epoch_loss_avg.result(),
                                                epoch_accuracy.result()))
    
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

plt.show()


test_dataset = tf.data.TextLineDataset(data_source)
test_dataset = test_dataset.skip(1)             # skip header row
test_dataset = test_dataset.map(parse_csv)      # parse each row with the function created earlier
test_dataset = test_dataset.shuffle(1000)       # randomize
test_dataset = test_dataset.batch(32)           # use the same batch size as the training set

# Unlike the training stage, the model only evaluates a single epoch of the test data. 
# In the following code cell, we iterate over each example in the test set and 
# compare the model's prediction against the actual label. 
# This is used to measure the model's accuracy across the entire test set.

test_accuracy = tfe.metrics.Accuracy()

for (x, y) in tfe.Iterator(test_dataset):
  prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))