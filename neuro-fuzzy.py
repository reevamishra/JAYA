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

#find the fuzzified inputs
def chebyshev(inputs):
	cheby_inputs = []
	i = 0
	while i<7:
		val = inputs[i]
		cheby_inputs.append(val)
		val = ((2*inputs[i]*inputs[i]) - 1)
		cheby_inputs.append(val)
		val = ((4*inputs[i]*inputs[i]*inputs[i]) - (3*inputs[i]))
		cheby_inputs.append(val)
		i+=1
	return cheby_inputs

#network initialization
def initialize_weights(n_inputs):
	weights = [random() for i in range(n_inputs)] 
	return weights

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

#train center and spread of ith membership for jth input and weights
total_iterations = 26
cheby_matrix_rows = 7
cheby_matrix_cols = 3

weights_matrix_rows = 3
weights_matrix_cols = 21
weights_matrix_hms = 20
weights_matrix = [[[[] for j in range(weights_matrix_cols)] for i in range(weights_matrix_rows)] for k in range(weights_matrix_hms)]
weights_matrix_new = weights_matrix

eta_matrix_cols = 6
eta_matrix_hms = weights_matrix_hms
eta_matrix = [[[] for i in range(eta_matrix_cols)] for k in range(eta_matrix_hms)]
eta_matrix_new = eta_matrix

hmcr = 0.7
par = 0.45
bw = exp(-6)
u_b = 1
l_b = 0

lambda1 = randint(0,100)/100.00
lambda2 = randint(0,100)/100.00
lambda3 = randint(0,100)/100.00

#...........................initialize weights...............................
seed(1)
weights_array = initialize_weights(weights_matrix_rows*weights_matrix_cols*weights_matrix_hms)	


#...............form weights matrix........................................
n=0
while n<(weights_matrix_rows*weights_matrix_cols*weights_matrix_hms):
	for k in range(weights_matrix_hms):
		for i in range(weights_matrix_rows):
			for j in range(weights_matrix_cols):
				weights_matrix[k][i][j] = weights_array[n]
				n += 1	

#...........................initialize eta weights...............................
seed(1)
eta_array = initialize_weights(eta_matrix_cols*eta_matrix_hms)	


#...............form weights matrix........................................
n=0
while n<(eta_matrix_cols*eta_matrix_hms):
	for k in range(eta_matrix_hms):
		for i in range(eta_matrix_cols):
			eta_matrix[k][i] = eta_array[n]
			n += 1	

#...............train the network for 6 times.................................... 
outputs_at_memory = []
temporal_firing_strength_memory = [[[] for j in range(3)] for i in range(20)]


#...................Training for first 6 inputs..............
t=1501
while t<=1520:

	#...............form the inputs array(crisp = 7, cheby_inputs=21, fuzzy_cheby_inputs=21)...................
	avg = (z[t] + z[t-1] + z[t-2] + z[t-3] + z[t-4] + z[t-7])/6
	inputs = [z[t], z[t-1], z[t-2], z[t-3], z[t-4], z[t-7], avg]
	inputs_cheby_array = chebyshev(inputs)
	cheby_inputs = [[[] for j in range(cheby_matrix_cols)] for i in range(cheby_matrix_rows)]

	#....................form crisp cheby input matrix...............................
	k= 0
	while k<21:
		for i in range(cheby_matrix_rows):
			for j in range(cheby_matrix_cols):
				cheby_inputs[i][j] = inputs_cheby_array[k]
				k += 1

	#....................create fuzzify cheby inputs.................................
	cheby_nodes_sum_array = []
	center_of_fuzzy_set = []
	width_fuzzy_set = [[[] for j in range(cheby_matrix_cols)] for i in range(cheby_matrix_rows)]
	fuzzy_inputs = [[[] for j in range(cheby_matrix_cols)] for i in range(cheby_matrix_rows)]
	
	for i in range(cheby_matrix_rows):   #.................find center of fuzzy set.............
		cheby_nodes_sum = 0
		for j in range(cheby_matrix_cols):
			cheby_nodes_sum += cheby_inputs[i][j]
		center_of_fuzzy_set.append(cheby_nodes_sum/cheby_matrix_rows)

	for i in range(cheby_matrix_rows):  #.................find width of fuzzy set...............
			for j in range(cheby_matrix_cols):
				width_fuzzy_set[i][j] = (cheby_inputs[i][j]-center_of_fuzzy_set[i])/cheby_matrix_rows

	for i in range(cheby_matrix_rows):  #...............find out fuzzy inputs from center and width of fuzzy sets...........
			for j in range(cheby_matrix_cols):
				fuzzy_inputs[i][j] = exp(  ((-1)*(cheby_inputs[i][j] - center_of_fuzzy_set[i])*(cheby_inputs[i][j] - center_of_fuzzy_set[i]))  /  (2*width_fuzzy_set[i][j]*width_fuzzy_set[i][j]) )


	#...........select the minimum of the top, middle, lower values........
	top = []
	middle = []
	bottom = []
	for i in range(cheby_matrix_rows):
			for j in range(cheby_matrix_cols):
				if j==0:
					top.append(fuzzy_inputs[i][j])
				if j==1:
					middle.append(fuzzy_inputs[i][j])
				if j==2:
					bottom.append(fuzzy_inputs[i][j])
	top_val = min(top)
	middle_val = min(middle)
	bottom_val = min(bottom)

	

	



	if (t<=1506):

		#.............calculate current and temporal firing strength.................
		current_firing_strength_1 = top_val/(top_val + middle_val + bottom_val)
		current_firing_strength_2 = middle_val/(top_val + middle_val + bottom_val)
		current_firing_strength_3 = bottom_val/(top_val + middle_val + bottom_val)

		temporal_firing_strength_1 =  (lambda1*current_firing_strength_1) + ( (1-lambda1) * 0)
		temporal_firing_strength_2 =  (lambda2*current_firing_strength_2) + ( (1-lambda2) * 0)
		temporal_firing_strength_3 =  (lambda3*current_firing_strength_3) + ( (1-lambda3) * 0)

		temporal_firing_strength_memory[t-1501][0] = temporal_firing_strength_1
		temporal_firing_strength_memory[t-1501][1] = temporal_firing_strength_2
		temporal_firing_strength_memory[t-1501][2] = temporal_firing_strength_3


		#...............produce the first set of f(j) values.......................
		k = 0 
		raw_outputs = [[[] for j in range(weights_matrix_rows)] for i in range(weights_matrix_hms)]
		while k<weights_matrix_hms:
			i=0
			while i<weights_matrix_rows:
				j=0
				outputs = 0
				while j<weights_matrix_cols:
					outputs += weights_matrix[k][i][j]*inputs_cheby_array[j]
					j+=1
				raw_outputs[k][i] = tanh(outputs)
				i+=1
			k+=1 

		#...............array to save the outputs of initial set of weights set...........
		outputs = [[] for i in range(weights_matrix_hms)]

		#................produce the initial final outputs.................................
		i = 0
		while i<weights_matrix_hms:
			j=0
			final_outputs = 0
			while j<weights_matrix_rows:
				final_outputs += temporal_firing_strength_1*raw_outputs[i][j] + temporal_firing_strength_2*raw_outputs[i][j+1] + temporal_firing_strength_3*raw_outputs[i][j+2] 
				j+=3
			outputs[i] = final_outputs
			i+=1

		desired = z[t+1]

		#................calculate error for each weight set........................
		error_array = []
		for i in range(weights_matrix_hms):
			error = outputs[i] - desired
			error_array.append(error*error)


		#................modify old weights by HS algorithm to generate new weights.............................
		for k in range(weights_matrix_hms):
			r_1 = randint(0,100)/100.00
			r_2 = randint(0,100)/100.00
			r_3 = randint(0,100)/100.00
			for i in range(weights_matrix_rows):
				for j in range(weights_matrix_cols):
					if(r_1<hmcr):
						weights_matrix_new[k][i][j] = weights_matrix[k][i][j]
						if(r_2<par):
							weights_matrix_new[k][i][j] = weights_matrix_new[k][i][j] - (r_3*bw)
						if(weights_matrix_new[k][i][j]<l_b):
							weights_matrix_new[k][i][j] = l_b
						if(weights_matrix_new[k][i][j]>u_b):
							weights_matrix_new[k][i][j] = u_b
					else:
						weights_matrix_new[k][i][j] = l_b + random()*(u_b - l_b)


		#...............produce the new set of f(j) values.......................
		k = 0 
		raw_outputs_new = [[[] for j in range(weights_matrix_rows)] for i in range(weights_matrix_hms)]
		while k<weights_matrix_hms:
			i=0
			while i<weights_matrix_rows:
				j=0
				outputs_new = 0
				while j<weights_matrix_cols:
					outputs_new += weights_matrix[k][i][j]*inputs_cheby_array[j]
					j+=1
				raw_outputs_new[k][i] = tanh(outputs_new)
				i+=1
			k+=1

		#...............array to save the outputs of initial set of weights set...........
		outputs_new = outputs
		#................produce the new final outputs(with new weights)...................
		i = 0
		while i<weights_matrix_hms:
			j=0
			final_outputs_new = 0
			while j<weights_matrix_rows:
				final_outputs_new += temporal_firing_strength_1*raw_outputs_new[i][j] + temporal_firing_strength_2*raw_outputs_new[i][j+1] + temporal_firing_strength_3*raw_outputs_new[i][j+2] 
				j+=3
			outputs_new[i] = final_outputs_new
			i+=1

		desired = z[t+1]

		#................calculate error for each weight set........................
		error_array_new = []
		for i in range(weights_matrix_hms):
			error_new = outputs_new[i] - desired
			error_array_new.append(error_new*error_new)

		#...............select the set of weights set by Jaya algorithm.............
		for i in range(weights_matrix_hms):
			if(error_array_new[i]<error_array[i]):
				error_array[i] = error_array_new[i]
				weights_matrix[i] = weights_matrix_new[i]
				outputs[i] = outputs_new[i]

		#Now, we got a set of 20 updated weights set that will be fed as input to next training generation. From the old and new weights set, we selected the weights set that generate least error square by Jaya.
		#Also select the output at memory with the least error square value....
		minimum = min(error_array)
		m=0
		while m<weights_matrix_hms:
			if error_array[m] == minimum:
				break
			m+=1

		outputs_at_memory.append(outputs[m])

	t += 1






	if (t>=1507):

		#.............calculate current and temporal firing strength.................
		current_firing_strength_1 = top_val/(top_val + middle_val + bottom_val)
		current_firing_strength_2 = middle_val/(top_val + middle_val + bottom_val)
		current_firing_strength_3 = bottom_val/(top_val + middle_val + bottom_val)

		temporal_firing_strength_1 =  (lambda1*current_firing_strength_1) + ( (1-lambda1) * 0)
		temporal_firing_strength_2 =  (lambda2*current_firing_strength_2) + ( (1-lambda2) * 0)
		temporal_firing_strength_3 =  (lambda3*current_firing_strength_3) + ( (1-lambda3) * 0)

		temporal_firing_strength_memory[t-1501][0] = temporal_firing_strength_1
		temporal_firing_strength_memory[t-1501][1] = temporal_firing_strength_2
		temporal_firing_strength_memory[t-1501][2] = temporal_firing_strength_3

		#...............produce the first set of f(j) values.......................
		k = 0 
		raw_outputs = [[[] for j in range(weights_matrix_rows)] for i in range(weights_matrix_hms)]
		while k<weights_matrix_hms:
			i=0
			while i<weights_matrix_rows:
				j=0
				outputs = 0
				while j<weights_matrix_cols:
					outputs += weights_matrix[k][i][j]*inputs_cheby_array[j]
					j+=1
				raw_outputs[k][i] = tanh(outputs)
				i+=1
			k+=1 

		#.......array to save the outputs of initial set of outputs set..........
		outputs = [[] for i in range(weights_matrix_hms)]

		#.......array to save the outputs of initial set of outputs set by considering upto 6 steps delayed outputs..........
		outputs_delayed = [[] for i in range(weights_matrix_hms)]
		#.......sum of current and delayed inputs outputs..........
		outputs_final = [[] for i in range(weights_matrix_hms)]

		#.......produce the initial final outputs................................
		i = 0
		while i<weights_matrix_hms:
			j=0
			final_outputs = 0
			while j<weights_matrix_rows:
				final_outputs += temporal_firing_strength_1*raw_outputs[i][j] + temporal_firing_strength_2*raw_outputs[i][j+1] + temporal_firing_strength_3*raw_outputs[i][j+2] 
				j+=3
			outputs[i] = final_outputs
			i+=1

		#........produce the initial final outputs considering upto 6 steps delayed outputs......
		i = 0  
		k = t-1501-1 #...............-1 bcoz array starts from 0.................................
		while i<20:  
			final_outputs = 0
			final_outputs += outputs_at_memory[k] * eta_matrix[i][5] 
			final_outputs += outputs_at_memory[k-1] * eta_matrix[i][4] 
			final_outputs += outputs_at_memory[k-2] * eta_matrix[i][3] 
			final_outputs += outputs_at_memory[k-3] * eta_matrix[i][2] 
			final_outputs += outputs_at_memory[k-4] * eta_matrix[i][1] 
			final_outputs += outputs_at_memory[k-5] * eta_matrix[i][0] 
			outputs_delayed[i] = final_outputs
			i+=1

		#..... calculate sum of outputs and outputs_delayed........................
		for i in range(weights_matrix_hms):
			outputs_final[i] = outputs[i] + outputs_delayed[i]

		desired = z[t+1]

		#................calculate error for each weight set........................
		error_array = []
		for i in range(weights_matrix_hms):
			error = outputs_final[i] - desired
			error_array.append(error*error)


		#................modify old weights by HS algorithm to generate new weights.............................
		for k in range(weights_matrix_hms):
			r_1 = randint(0,100)/100.00
			r_2 = randint(0,100)/100.00
			r_3 = randint(0,100)/100.00
			for i in range(weights_matrix_rows):
				for j in range(weights_matrix_cols):
					if(r_1<hmcr):
						weights_matrix_new[k][i][j] = weights_matrix[k][i][j]
						if(r_2<par):
							weights_matrix_new[k][i][j] = weights_matrix_new[k][i][j] - (r_3*bw)
						if(weights_matrix_new[k][i][j]<l_b):
							weights_matrix_new[k][i][j] = l_b
						if(weights_matrix_new[k][i][j]>u_b):
							weights_matrix_new[k][i][j] = u_b
					else:
						weights_matrix_new[k][i][j] = l_b + random()*(u_b - l_b)


		#...............produce the new set of f(j) values.......................
		k = 0 
		raw_outputs_new = [[[] for j in range(weights_matrix_rows)] for i in range(weights_matrix_hms)]
		while k<weights_matrix_hms:
			i=0
			while i<weights_matrix_rows:
				j=0
				final_outputs_new = 0
				while j<weights_matrix_cols:
					final_outputs_new += weights_matrix_new[k][i][j]*inputs_cheby_array[j]
					j+=1
				raw_outputs_new[k][i] = tanh(outputs_new)
				i+=1
			k+=1

		#...............array to save the outputs of initial set of weights set............
		outputs_new = outputs

		#.......array to save the outputs of initial set of outputs set by considering upto 6 steps delayed outputs..........
		outputs_delayed_new = [[] for i in range(weights_matrix_hms)]
		#.......sum of current and delayed inputs outputs..........
		outputs_final_new = [[] for i in range(weights_matrix_hms)]

		#................produce the new final outputs(with new weights)...................
		i = 0
		while i<weights_matrix_hms:
			j=0
			final_outputs_new = 0
			while j<weights_matrix_rows:
				final_outputs_new += temporal_firing_strength_1*raw_outputs_new[i][j] + temporal_firing_strength_2*raw_outputs_new[i][j+1] + temporal_firing_strength_3*raw_outputs_new[i][j+2] 
				j+=3
			outputs_new[i] = final_outputs_new
			i+=1


		#........produce the new final outputs considering upto 6 steps delayed outputs......
		i = 0
		while i<eta_matrix_hms:
			j=t-1501
			final_outputs_new = 0
			while j>=0:
				final_outputs_new += eta_matrix_new[i][j-1] * outputs_at_memory[j-1] 
				j-=1
			outputs_delayed_new[i] = final_outputs_new
			i+=1

		#..... calculate sum of outputs and outputs_delayed........................
		for i in range(weights_matrix_hms):
			outputs_final_new[i] = outputs_new[i] + outputs_delayed_new[i]

		desired = z[t+1]

		#................calculate error for each weight set........................
		error_array_new = []
		for i in range(weights_matrix_hms):
			error_new = outputs_final_new[i] - desired
			error_array_new.append(error_new*error_new)

		#...............select the set of weights set by Jaya algorithm.............
		for i in range(weights_matrix_hms):
			if(error_array_new[i]<error_array[i]):
				error_array[i] = error_array_new[i]
				weights_matrix[i] = weights_matrix_new[i]
				eta_matrix[i] = eta_matrix_new[i]
				outputs[i] = outputs_new[i]

		#Now, we got a set of 20 updated weights set that will be fed as input to next training generation. From the old and new weights set, we selected the weights set that generate least error square by Jaya.
		#Also select the output at memory with the least error square value....
		minimum = min(error_array)
		m=0
		while m<weights_matrix_hms:
			if error_array[m] == minimum:
				break
			m+=1

		outputs_at_memory.append(outputs[m])

	t+=1



