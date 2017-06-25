# Backprop on the Seeds Dataset
import csv
import random as rand
from math import exp
import numpy as np
import matplotlib.pyplot as plt
# Load a CSV file
def load_csv(filename):
    ifile=open(filename, "r")
    reader=csv.reader(ifile)
    inputs=[]
    for row in reader:
        inputs1=[]
        for i in range(len(row)-2):
            inputs1.append(float(row[i]))
        inputs1.append(int(row[len(row)-1]))
        inputs.append(inputs1)
    ifile.close()
    return inputs
# Find the min and max values for each column
def findminmax(dataset):
	minmax = []
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# Rescale dataset columns to the range 0-1
def normalize(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
def normalizesample(row,minmax):
    for i in range(len(row)-1):
        row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
# Split a dataset into k Sets
def split_data(dataset, n_set):
	dataset_split = []
	temp = dataset
	set_size = int(len(dataset)/n_set)
	for i in range(n_set):
		data = []
		while len(data) < set_size:
			index =rand.randrange(len(temp))
			data.append(temp.pop(index))
		dataset_split.append(data)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_cal(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
    
# Evaluate an algorithm using a cross validation split
def evaluate(dataset, algorithm, n_set,l_rate, n_epoch, n_hidden):  
    split_set = split_data(dataset, n_set)
    scores = []
    count=0
    print(len(split_set))
    for data in split_set:
        train_set = list(split_set)
        train_set.remove(data)
        train_set = sum(train_set, [])
        test_set = []
        for row in data:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, l_rate, n_epoch, n_hidden)[0]
        actual = [row[-1] for row in data]
        accuracy = accuracy_cal(actual, predicted)
        scores.append(accuracy)
    print(count,'count')
    return scores
        
#Predict if a sample is given
def predictsample(dataset, algorithm, n_set,l_rate, n_epoch, n_hidden,testsample):
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, dataset, l_rate, n_epoch, n_outputs)
    return predict(network, row)
    
# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation
        
# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, inputs):
    for i in network:
        nextinputs=[]
        for j in i:
            activ=activate(j['weights'],inputs)
            functionaleval=transfer(activ)
            j['output']=functionaleval
            j['activation']=activ
            nextinputs.append(functionaleval)
        inputs=nextinputs
    return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                rval=expected[j] - neuron['output']
                errors.append(rval)
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
    return rval
#Updates the Weights Accordingly
def update_weights(network,inputs,n):
    for layer in range(len(network)):
        if(layer!=0):
            inputs=[]
            prevlayer=network[layer-1]
            for node in prevlayer:
                inputs.append(node['output'])
            inputs.append(1)
        ly=network[layer]
        for node in range(len(ly)):
            neuron=ly[node]
            for weight in range(len(neuron['weights'])-1):
                neuron['weights'][weight]+=n*neuron['delta']*inputs[weight]
            neuron['weights'][len(neuron['weights'])-1]+=n*neuron['delta']
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    delta=list()
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]-1] = 1
            error=backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
#        print('>epoch=%d, lrate=%.3f' % (epoch, l_rate), network[-1][0]['delta'])
        delta.append(error)
    fig=plt.figure()
    plt.plot(delta)
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = []
    hidden_layer=[]
    output_layer=[]
    for i in range(n_hidden):
        w=[]
        for i in range(n_inputs + 1):
            w.append(rand.random())
            wd={'weights':w}
        hidden_layer.append(wd)
    network.append(hidden_layer)
    for i in range(n_outputs):
        w=[]
        for i in range(n_hidden + 1):
            w.append(rand.random())
            wd={'weights':w}
        output_layer.append(wd)
    network.append(output_layer)
    return network
    
# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))+1
    
# Backpropagation Algorithm With Stochastic Gradient Descent

def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = []
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    relis=[]
    relis.append(predictions)
    relis.append(network)
    return(relis)

# Test Backprop on Seeds dataset

# load and prepare data
filename = 'seeds_dataset.csv'
dataset = load_csv(filename)
# normalize input variables
minmax = findminmax(dataset)
normalize(dataset, minmax)

a=input('Press 1 for Evaluation \nPress 2 for Prediction\n')
if(a==1):
    # evaluate algorithm
    n_set = 5
    l_rate = 0.3
    n_epoch = 750
    n_hidden = 5
    scores = evaluate(dataset, back_propagation, n_set, l_rate, n_epoch, n_hidden)
    print('Scores: ' , scores)
    print('Mean Accuracy:' ,(sum(scores)/float(len(scores))))
elif(a==2):
    inputs=[16.44,15.25,0.888,5.884,3.505,1.969,5.533]
    n_set = 5
    l_rate = 0.3
    n_epoch = 750
    n_hidden = 5
    ans = predictsample(dataset, back_propagation, n_set, l_rate, n_epoch, n_hidden,inputs)
    print(ans)
