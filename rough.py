from random import seed
from random import random
from random import randint
from math import cos, sin, exp, pi, tanh, sqrt
from csv import reader
import numpy as np
import matplotlib.pyplot as plt

weights_matrix_rows = 3
weights_matrix_cols = 2
weights_matrix_hms = 5

k = 0 
raw_outputs = [[[] for j in range(weights_matrix_rows)] for i in range(weights_matrix_hms)]
while k<weights_matrix_hms:
	i=0
	while i<weights_matrix_rows:
		j=0
		outputs = 0
		while j<weights_matrix_cols:
			outputs += i+j
			j+=1
		activation = outputs
		raw_outputs[k][i] = activation
		i+=1
	k+=1
print(raw_outputs)

k=1
while k <=6:
	final_outputs_new += eta_matrix_new[i][t - 1501 - k] * outputs_at_memory[t-1501-k]
	k+=1

k=1
while k <=6:
	final_outputs_new += eta_matrix_new[i][t - 1501 - k] * outputs_at_memory[t-1501-k]
	k+=1