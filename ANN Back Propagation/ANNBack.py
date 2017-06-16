# Backprop on the Seeds Dataset
import csv
import random as rand
from math import exp
# Load a CSV file
def load_csv(filename):
    ifile=open('seeds_dataset.csv', "r")
    reader=csv.reader(ifile)
    inputs=[]
    for row in reader:
        inputs1=[]
        inputs1.append(float(row[0]))
        inputs1.append(float(row[1]))
        inputs1.append(float(row[2]))
        inputs1.append(float(row[3]))
        inputs1.append(float(row[4]))
        inputs1.append(float(row[5]))
        inputs1.append(float(row[6]))
        inputs1.append(int(row[7]))
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

# Split a dataset into k Sets
def cross_validation_split(dataset, n_set):
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
	correct = 0;print('predicted:',predicted);print('Actual:',actual)
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate(dataset, algorithm, n_set, *args):
	split_set = cross_validation_split(dataset, n_set)
	scores = []
	for data in split_set:
		train_set = split_set
		train_set.remove(data)
		train_set = sum(train_set, [])
		test_set = []
		for row in data:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in data]
		accuracy = accuracy_cal(actual, predicted)
		scores.append(accuracy)
	return scores
		
# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-2):
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
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

#Updates the Weights Accordingly
def update_weights(network,inputs,n):
    for layer in range(len(network)):
        if(layer==0):
            ly=network[layer]
            for node in range(len(ly)):
                neuron=ly[node]
                for weight in range(len(neuron['weights'])-1):
                    neuron['weights'][weight]+=n*neuron['delta']*inputs[weight]
                neuron['weights'][len(neuron['weights'])-1]+=n*neuron['delta']
        else:
            inputs=[]
            prevlayer=network[layer-1]
            for node in prevlayer:
                inputs.append(node['output'])
            inputs.append(1)
            ly=network[layer]
            for node in range(len(ly)):
                neuron=ly[node]
                for weight in range(len(neuron['weights'])):
                    neuron['weights'][weight]+=n*neuron['delta']*inputs[weight]
                neuron['weights'][len(neuron['weights'])-1]+=n*neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]-1] = 1
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)

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
	return(predictions)

# Test Backprop on Seeds dataset

# load and prepare data
filename = 'seeds_dataset.csv'
dataset = load_csv(filename)
print(dataset[100])
# normalize input variables
minmax = findminmax(dataset)
normalize(dataset, minmax)

# evaluate algorithm
n_set = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 5
scores = evaluate(dataset, back_propagation, n_set, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
