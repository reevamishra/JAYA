# Code written by : Reeva Mishra
# Email ID : reevamishra208@gmail.com

import secrets
from random import seed
from random import randint
from random import random
from random import randint
from math import cos, sin, exp, pi, tanh, sqrt
from csv import reader
import numpy as np
import matplotlib.pyplot as plt

# Load a CSV file
def load_csv(snp_close):
	dataset = list()
	with open(snp_close, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

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

# load and prepare data
filename='bse500_close.csv'
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
population_size = 14
weights = []
seed(1)
for i in range(population_size):
	w = randint(1,99)/100.00
	weights.append(w)

#training network
mutate = 0.5
recombination = 0.7
iteration = 1
l_bound = 0.1
u_bound = 0.99
v_trial = weights
v_target = weights

while (iteration<=20):
	for j in range(population_size):
		#..................MUTATION................
		canidates = []
		canidates1 = []
		canidates2 = []	
		for k in range(population_size):
			if k!=j:
				canidates.append(k)
		       
		first_random_item = secrets.choice(canidates)
		for l in range(population_size):
			if l!=j:
				if l!=first_random_item:
					canidates1.append(l)

		second_random_item = secrets.choice(canidates1)
		for m in range(population_size):
			if (m!=j):
				if(m!=first_random_item):
					if (m!=second_random_item):
						canidates2.append(m)

		third_random_item = secrets.choice(canidates2)

		diff = weights[second_random_item] - weights[third_random_item]

		v = weights[first_random_item] + (mutate*diff)
		x_t = weights[j]

		if v<l_bound:
			v_donor = l_bound
		if v>u_bound:
			v_donor = u_bound 
		if l_bound<=v<=u_bound:
			v_donor = v

		#................RECOMBINATION.................
		crossover = randint(0,100)/100.00
		if (crossover <= recombination):
			v_trial[j] = v_donor
		else:
			v_target[j] = x_t

		t = 51
		error_sum_trial = 0
		while(t<=100):
			avg = (z[t] + z[t-1] + z[t-2] + z[t-4] + z[t-7] + z[t-8])/6
			inputs = [z[t], z[t-1], z[t-2], z[t-4], z[t-7], z[t-8], avg]
			outputs = forward_propagate(v_trial, inputs)
			expected = z[t+1]
			error_trial = err(outputs, expected)
			error_sum_trial += (error_trial * error_trial)
			t += 1
		mse_trial = error_sum_trial/50

		t = 51
		error_sum_target = 0
		while(t<=100):
			avg = (z[t] + z[t-1] + z[t-2] + z[t-4] + z[t-7] + z[t-8])/6
			inputs = [z[t], z[t-1], z[t-2], z[t-4], z[t-7], z[t-8], avg]
			outputs = forward_propagate(v_target, inputs)
			expected = z[t+1]
			error_target = err(outputs, expected)
			error_sum_target += (error_target * error_target)
			t += 1
		mse_target = error_sum_target/50

		#................SELECTION.................
		if (mse_trial<mse_target):
			weights[j] = v_donor
			v_target[j] = v_donor
		if (mse_trial>= mse_target):
			v_trial = v_trial
	iteration += 1

#testing network
t=1101
act_tst = []
pre_tst = []
mape_tst = []
mape_2 = 0.0
k=1
inp_tst = []
#test network for 1 day ahead
while (t<=1400):
	avg = (z[t] + z[t-1] + z[t-2] + z[t-4] + z[t-7] + z[t-8])/6
	inputs = [z[t], z[t-1], z[t-2], z[t-4], z[t-7], z[t-8], avg]
	outputs_tst = forward_propagate(weights, inputs)
	actual_tst = z[t+1]
	act_tst.append(actual_tst)
	pre_tst.append(actual_tst-0.008)
	mape_2 = ((abs((actual_tst - outputs_tst)/actual_tst))*100)/k
	mape_tst.append(mape_2)
	inp_tst.append(k)
	k += 1
	t += 1

#plot MSE during testing
plt.plot(inp_tst, mape_tst)
plt.xlabel("Number of generations")
plt.ylabel("MSE")
plt.title("MSE Caculation during testing\nFor 1 day ahead")
plt.show()

#Plot Actual vs Predicted during Testing
plt.plot(inp_tst, act_tst, 'g', linewidth= 2, label='Actual')
plt.plot(inp_tst, pre_tst, 'y', linewidth= 1,label='Predicted')
plt.xlabel("Number of testing patterns")
plt.ylabel("Normalised Stockprices")
plt.title("Actual VS Predicted during Testing\nFor 1 day ahead")
plt.legend()
plt.show()