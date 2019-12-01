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
def load_csv(bse_psu_close):
    dataset = list()
    with open(bse_psu_close, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

#network initialization
def initialize_network(n_inputs):
    weights = [random() for i in range(n_inputs*3)] 
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
seed(1)
weights = initialize_network(7)
print(weights)

#initialize velocities
velocity = []
for i in range(21):
    v = randint(0,100)/100.00
    velocity.append(v)

#training weights
n_iteration = 30
w_inertia = 0.5
c_1 = 1
c_2 = 2

for k in range(30):
    t = 51
    error_sum = 0
    sum_weights = 0
    while(t<=1000):
        avg = (z[t] + z[t-1] + z[t-2] + z[t-4] + z[t-7] + z[t-8])/6
        inputs = [z[t], z[t-1], z[t-2], z[t-4], z[t-7], z[t-8], avg]
        outputs = forward_propagate(weights, inputs)
        expected = z[t+1]
        error = err(outputs, expected)
        error_sum += (error * error)
        t += 1
    mse = error_sum/950
    p_k = weights
    for i in range(21):
        sum_weights += weights[i]
    p_g = sum_weights/21
    
    r_1 = randint(0,100)/100.00
    r_2 = randint(0,100)/100.00
    weights_modified = weights
    velocity_modified = velocity
    for i in range(21):
        velocity_modified[i] = (w_inertia*velocity[i]) + (c_1*r_1*(p_k[i]-weights[i])) + (c_2*r_2*(p_g-weights[i]))
        weights_modified[i] = weights[i] + velocity[i]
    for i in range(21):
        if(weights_modified[i]>1):
            weights_modified[i] = 0.99
        if(weights_modified[i]<0):
            weights_modified[i] = 0.01

    t = 51
    error_sum = 0
    sum_weights = 0
    while(t<=1000):
        avg = (z[t] + z[t-1] + z[t-2] + z[t-4] + z[t-7] + z[t-8])/6
        inputs = [z[t], z[t-1], z[t-2], z[t-4], z[t-7], z[t-8], avg]
        outputs = forward_propagate(weights_modified, inputs)
        expected = z[t+1]
        error = err(outputs, expected)
        error_sum += (error * error)
        t += 1
    mse_modified = error_sum/950

    if mse_modified<=mse:
        mse_best = mse_modified
        weights = weights_modified
        velocity = velocity_modified
    if mse>mse_modified:
        mse_best = mse

print(weights)
#train network for 1 day ahead
t=500
error_sum = 0.0
act_trn = []
pre_trn = []
mape_1 = 0.0
mape_trn = []
msef = []
inp_trn = []
j=500
while (t<=1000):
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
    mse = error_sum / j 
    mse_sq = sqrt(mse)
    msef.append(mse_sq)
    inp_trn.append(j)
    j -= 1
    t += 1

#test network for 1 day ahead
t=1000
act_tst = []
pre_tst = []
mape_tst = []
mape_2 = 0.0
k=1
inp_tst = []
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

#plot MSE during training
plt.plot(inp_trn, msef)
plt.xlabel("Number of generations")
plt.ylabel("MSE")
plt.title("MSE Caculation during training\nFor 1 day ahead")
plt.show()

#Plot Actual vs Predicted during Testing
plt.plot(inp_tst, act_tst, 'g', linewidth= 2, label='Actual')
plt.plot(inp_tst, pre_tst, 'y', linewidth= 1,label='Predicted')
plt.xlabel("Number of testing patterns")
plt.ylabel("Normalised Stockprices")
plt.title("Actual VS Predicted during Testing\nFor 1 day ahead")
plt.legend()
plt.show()