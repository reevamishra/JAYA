# Code written by : Reeva Mishra
# Email ID : reevamishra208@gmail.com

from random import seed
from random import random
from math import cos, sin, exp, pi, tanh, sqrt
from csv import reader
import numpy as np
import matplotlib.pyplot as plt

# Load a CSV file
def load_csv(nasdaq_close):
	dataset = list()
	with open(nasdaq_close, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

#network initialization
def initialize_network(n_inputs):
	weights = [random() for i in range(n_inputs*2)] 
	return weights


#produce the first dataset
def activate(weights, inputs):
	activation = 0.0
	activates = []
	i=0
	j=0
	k=0
	while i<7:
		a = weights[i]*cos(pi*inputs[j])
		activation += a
		a = weights[i+1]*sin(pi*inputs[j])
		activation += a
		i+=2
		j+=1
	return activation

#Neuron activation transfer function
def transfer(activation):
	return tanh(activation)
 
#forward propagation
def forward_propagate(weights, inputs):
	w = weights
	i = inputs
	activation = activate(w, i)
	outputs = transfer(activation)
	return outputs

#calculate error
def err(outputs, expected):
	error = outputs - expected
	return error

#transfer function
def transfer_derivative(outputs):
	return (1 -  (outputs*outputs))

#update weights
def update_weights(weights, l_rate, error, outputs, inputs):
	w_update = weights
	new_weights = []
	j=0
	i=0
	while j<7:
		w = weights[i] - (l_rate * error * transfer_derivative(outputs) * inputs[j])
		new_weights.append(w)
		w = weights[i+1] - (l_rate * error * transfer_derivative(outputs) * inputs[j])
		new_weights.append(w)
		j+=1
		i+=2
	w_update = new_weights
	return w_update

# load and prepare data
filename='nasdaq_close.csv'
dataset = load_csv(filename)
x = np.array(dataset)
y = x.astype(np.float)
n = len(y)
#normalize inputs
y_min = min(y)
y_max = max(y)
z = []
for i in range(n):
	norm = (y[i] - y_min) / (y_max - y_min)
	z.append(norm)

#initialize weights
seed(1)
weights = initialize_network(7)

l_rate=0.3
t=50
error_sum = 0.0
act_trn = []
pre_trn = []
mape_1 = 0.0
mape_trn = []
msef = []
mset = []
inp_trn = []
j=1

act_tst = []
pre_tst = []
mape_tst = []
mape_2 = 0.0
k=1
inp_tst = []
while(t<=1000):
	avg = (z[t] + z[t-1] + z[t-2] + z[t-4] + z[t-7] + z[t-8])/6
	inputs = [z[t], z[t-1], z[t-2], z[t-4], z[t-7], z[t-8], avg]
	outputs = forward_propagate(weights, inputs)
	expected = z[t+1]
	error = err(outputs, expected)
	error_sum += (error * error)
	weights = update_weights(weights, l_rate, error, outputs, inputs)
	t += 1

#train network for 1 day ahead
while (t<=1400):
	avg = (z[t] + z[t-1] + z[t-2] + z[t-4] + z[t-7] + z[t-8])/6
	inputs = [z[t], z[t-1], z[t-2], z[t-4], z[t-7], z[t-8], avg]
	outputs = forward_propagate(weights, inputs)
	actual = z[t+1]
	act_trn.append(actual)
	pre_trn.append(outputs)
	mape_1 = ((abs((actual - outputs)/actual))*100)/j
	mape_trn.append(mape_1)
	error = err(outputs, actual)
	error_sum += (error * error)
	weights = update_weights(weights, l_rate, error, outputs, inputs)
	mse = error_sum / j 
	mse_sq = sqrt(mse)
	msef.append(mse_sq)
	inp_trn.append(j)
	j += 1
	t += 1

t=1350
#test network for 1 day ahead
while (t<=1600):
	avg = (z[t] + z[t-1] + z[t-2] + z[t-4] + z[t-7] + z[t-8])/6
	inputs = [z[t], z[t-1], z[t-2], z[t-4], z[t-7], z[t-8], avg]
	outputs_tst = forward_propagate(weights, inputs)
	actual_tst = z[t+1]
	act_tst.append(actual_tst)
	pre_tst.append(outputs_tst)
	error_sq = (actual_tst - outputs_tst) * (actual_tst - outputs_tst)
	mset.append(error_sq/k)
	mape_2 = ((abs((actual_tst - outputs_tst)/actual_tst))*100)/k
	mape_tst.append(mape_2)
	inp_tst.append(k)
	k += 1
	t += 1

print(msef[399])
print(mape_1)
print(mset[249])
print(mape_2)

#plot MSE during training
plt.plot(inp_trn, msef)
plt.xlabel("Number of generations")
plt.ylabel("MSE")
plt.title("MSE Caculation during training\nFor 1 day ahead")
plt.show()

#plot MAPE during training
plt.plot(inp_trn, mape_trn)
plt.xlabel("Number of generations")
plt.ylabel("MAPE")
plt.title("MAPE Caculation during training\nFor 1 day ahead")
plt.show()

#Plot Actual vs Predicted during Training
plt.plot(inp_trn, act_trn, label='Actual')
plt.plot(inp_trn, pre_trn, label='Predicted')
plt.xlabel("Number of training patterns")
plt.ylabel("Normalised Stockprices")
plt.title("Actual VS Predicted during Training\nFor 1 day ahead")
plt.legend()
plt.show()

#plot MAPE during testing
plt.plot(inp_tst, mape_tst)
plt.xlabel("Number of generations")
plt.ylabel("MAPE")
plt.title("MAPE Caculation during testing\nFor 1 day ahead")
plt.show()

#Plot Actual vs Predicted during Testing
plt.plot(inp_tst, act_tst, label='Actual')
plt.plot(inp_tst, pre_tst, label='Predicted')
plt.xlabel("Number of training patterns")
plt.ylabel("Normalised Stockprices")
plt.title("Actual VS Predicted during Testing\nFor 1 day ahead")
plt.legend()
plt.show()