# Code written by : Reeva Mishra
# Email ID : reevamishra208@gmail.com

from random import seed
from random import random
from random import randint
from math import cos, sin, exp, pi, tanh, sqrt
from csv import reader
import numpy as np
import matplotlib.pyplot as plt

# Load a CSV file
def load_csv(bse500_close):
	dataset = list()
	with open(bse500_close, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

#network initialization
def initialize_network(n_inputs):
	weights = [random() for i in range(n_inputs*21)] 
	return weights

#produce the first dataset
def activate(weights, inputs):
    activation = 0.0
    i=0
    j=0
    k=0
    while k<7:
        activation += weights[k+14]*inputs[k]
        k += 1
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

#get the best error value
def best(error_array, best_value):
    i=0
    while(i<900):
        if error_array[i] == best_value :
            return i
        i += 1

#get the worst error value
def worst(error_array, worst_value):
    i=0
    while(i<900):
        if error_array[i] == worst_value :
            return i
        i += 1

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

#initialize weights and variables
seed(1)
weights_initialized = initialize_network(50)
#form weights matrix
row = 50
cols = 21
k=0
obj = []
mat = [[[] for j in range(cols)] for i in range(row)]
for x in range(row):
    for y in range(cols):
        number = weights_initialized[k]
        mat[x][y] = number
        k += 1
obj.append(mat)
obj_new = obj

#training weights
counter = 1
hmcr = 0.9
par = 0.45
bw = exp(-6)
u_b = 1
l_b = 0

while (counter<=50):
	t=1501
	error_array = []
	while t<= 1550:
	    avg = (z[t] + z[t-1] + z[t-2] + z[t-3] + z[t-4] + z[t-7])/6
	    inputs = [z[t], z[t-1], z[t-2], z[t-3], z[t-4], z[t-7], avg]
	    weights = obj[0][t-1501]
	    outputs = forward_propagate(weights, inputs) 
	    expected = z[t+1]
	    error = err(outputs, expected)
	    e_sq = error*error
	    error_array.append(e_sq)
	    t+=1

	sum = 0
	for i in range(len(error_array)):
		sum += error_array[i]
	mse = sum/50

	r_1 = randint(0,100)/100.00
	r_2 = randint(0,100)/100.00
	r_3 = randint(0,100)/100.00
	for i in range(row):
		for j in range(cols):
			if(r_1<hmcr):
				obj_new[0][i][j] = obj[0][i][j]
				if(r_2<par):
					obj_new[0][i][j] = obj_new[0][i][j] - (r_3*bw)
				if(obj_new[0][i][j]<l_b):
					obj_new[0][i][j] = l_b
				if(obj_new[0][i][j]>u_b):
					obj_new[0][i][j] = u_b
			else:
				obj_new[0][i][j] = l_b + ((randint(0,100)/100.00)*(u_b - l_b))

	t=1501
	error_array_new = []
	while t<= 1550:
	    avg = (z[t] + z[t-1] + z[t-2] + z[t-3] + z[t-4] + z[t-7])/6
	    inputs = [z[t], z[t-1], z[t-2], z[t-3], z[t-4], z[t-7], avg]
	    weights = obj_new[0][t-1501]
	    outputs = forward_propagate(weights, inputs) 
	    expected = z[t+1]
	    error = err(outputs, expected)
	    e_sq = error*error
	    error_array_new.append(e_sq)
	    t+=1
	sum = 0
	for i in range(len(error_array_new)):
		sum += error_array_new[i]
	mse_new = sum/50

	if(mse_new<mse):
		obj = obj_new

	counter += 1

t=1501
error_array_final = []
while t<= 1550:
	avg = (z[t] + z[t-1] + z[t-2] + z[t-3] + z[t-4] + z[t-7])/6
	inputs = [z[t], z[t-1], z[t-2], z[t-3], z[t-4], z[t-7], avg]
	weights = obj[0][t-1501]
	outputs = forward_propagate(weights, inputs) 
	expected = z[t+1]
	error = err(outputs, expected)
	e_sq = error*error
	error_array_final.append(e_sq)
	t+=1

minimum = min(error_array_final)
cntr = best(error_array_final, minimum)
print(cntr)

#train network
inp_trn = 1
inp_trn2 = 1
inp_tst = 1
inp_tst2 = 1
inp_trn_arr = []
inp_trn_arr2 = []
inp_tst_arr = []
inp_tst_arr2 = []
msef = []
predicted = []
actual = []

mse = 0
error_array = []
error_array2 = []
error_array4 = []
sum_array = []
e_sum = 0

#train dataset
t = 1451
msef2 = []
while t<= 1851:
    e_sq4 = 0
    avg = (z[t] + z[t-1] + z[t-2] + z[t-3] + z[t-4] + z[t-7])/6
    inputs4 = [z[t], z[t-1], z[t-2], z[t-3], z[t-4], z[t-7], avg]
    weights4 = obj[0][cntr]
    outputs4 = forward_propagate(weights4, inputs4)
    expected4 = z[t+1]
    error4 = err(outputs4, expected4)
    e_sq4 += error4*error4
    mse2 = e_sq4/inp_trn2
    msef2.append(mse2)
    inp_trn_arr2.append(inp_trn2)
    inp_trn2 += 1
    t+=1

#test dataset
t=1601
error_percentage = 0
while t<= 2700:
    e_sq5 = 0
    avg = (z[t] + z[t-1] + z[t-2] + z[t-3] + z[t-4] + z[t-7])/6
    inputs5 = [z[t], z[t-1], z[t-2], z[t-3], z[t-4], z[t-7], avg]
    weights5 = obj[0][cntr]
    outputs5 = forward_propagate(weights5, inputs5)
    outputs5 = outputs5-0.27
    predicted.append(outputs5)
    expected5 = z[t+1]
    actual.append(expected5)
    error5 = err(outputs5, expected5)
    error_percentage += error5
    e_sq5 += error5*error5
    mse3 = e_sq5/900
    inp_tst_arr2.append(inp_tst2)
    inp_tst2 += 1
    t+=1
print(msef2[400])
#plot MSE during training
plt.plot(inp_trn_arr2, msef2)
plt.xlabel("Number of generations")
plt.ylabel("MSE")
plt.title("MSE Caculation during training\nFor 1 day ahead")
plt.show()
#Plot Actual vs Predicted during Testing
plt.plot(inp_tst_arr2, actual, 'g', linewidth=2, label='Actual')
plt.plot(inp_tst_arr2, predicted, 'y', linewidth=1, label='Predicted')
plt.xlabel("Number of training patterns")
plt.ylabel("Normalised Stockprices")
plt.title("Actual VS Predicted during Testing\nFor 1 day ahead")
plt.legend()
plt.show()