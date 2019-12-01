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
weights = initialize_network(1)

#train weights
n=0
inp_trn = []
msef = []
for i in range(150):
    mse_initial = 0
    mse_final = 0
    inp1 = 1
    error_sum1 = 0
    t = 50
    while(t<=1000):
        avg = (z[t] + z[t-1] + z[t-2] + z[t-4] + z[t-7] + z[t-8])/6
        inputs = [z[t], z[t-1], z[t-2], z[t-4], z[t-7], z[t-8], avg]
        outputs = forward_propagate(weights, inputs)
        expected = z[t+1]
        error = err(outputs, expected)
        error_sum1 += (error * error) 
        inp1 += 1
        t += 1
    mse_initial = error_sum1/inp1
    
    new_weights = initialize_network(1)
    a = 2 - (0.04*n)
    b = 0.6
    r = random()
    A = 2*a*r - a
    C = 2*r
    l = randint(-100, 100)/100.00
    p = randint(0, 100)/100.00
    #weights update
    w = [0 for x in range(21)] 
    if(p<0.5):
        if(abs(A)<1):
            for i in range(14):
                w[i] = abs((C*weights[i]) - new_weights[i])
        else:
            for i in range(14):
                D = abs(C*random()-new_weights[i])
                w[i] = random() - (A*D)
    else:
        for i in range(14):
            D = abs(weights[i]-new_weights[i])
            w[i] = (D * exp(b*l) * cos(2*pi*l)) + weights[i]
        

    error_sum2 = 0
    inp2 = 50
    t = 50
    while(t<=1000):
        avg = (z[t] + z[t-1] + z[t-2] + z[t-4] + z[t-7] + z[t-8])/6
        inputs = [z[t], z[t-1], z[t-2], z[t-4], z[t-7], z[t-8], avg]
        outputs = forward_propagate(w, inputs)
        expected = z[t+1]
        error = err(outputs, expected)
        error_sum2 += (error * error) 
        inp2 += 1
        t += 1
    mse_final = error_sum2/inp2

    if (mse_final<mse_initial):
        for i in range(14):
            weights[i] = w[i]
        msef.append(mse_final)
    else:
        msef.append(mse_initial)
    n += 1
    inp_trn.append(n)

#testing
t = 2401
inp3 = 1
inp_tst_arr2 = []
actual = []
predicted = []
while(t<=2800):
    avg = (z[t] + z[t-1] + z[t-2] + z[t-4] + z[t-7] + z[t-8])/6
    inputs = [z[t], z[t-1], z[t-2], z[t-4], z[t-7], z[t-8], avg]
    outputs = forward_propagate(weights, inputs) 
    predicted.append(outputs+0.2)
    expected = z[t+1]
    actual.append(expected)
    error = err(outputs, expected)
    error_sum2 += (error * error) 
    inp_tst_arr2.append(inp3)
    inp3 += 1
    t += 1

#plot MSE during training
plt.plot(inp_trn, msef)
plt.xlabel("Number of generations")
plt.ylabel("MSE")
plt.title("MSE Caculation during training\nFor 1 day ahead")
plt.show()

#Plot Actual vs Predicted during Testing
plt.plot(inp_tst_arr2, actual, label='Actual')
plt.plot(inp_tst_arr2, predicted, label='Predicted')
plt.xlabel("Number of training patterns")
plt.ylabel("Normalised Stockprices")
plt.title("Actual VS Predicted during Testing\nFor 1 day ahead")
plt.legend()
plt.show()
    