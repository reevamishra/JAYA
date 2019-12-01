
	#.....................update 21 weights by harmony search..........................
	r_1 = randint(0,100)/100.00
	r_2 = randint(0,100)/100.00
	r_3 = randint(0,100)/100.00
	r_4 = randint(0,100)/100.00
	for i in range(3):
		for j in range(21):
			if(r_1<hmcr):
				weights_matrix_new[i][j] = weights_matrix[i][j]
				if(r_2<par):
					weights_matrix_new[i][j] = weights_matrix_new[i][j] - (r_3*bw)
				if(weights_matrix_new[i][j]<l_b):
					weights_matrix_new[i][j] = l_b
				if(weights_matrix_new[i][j]>u_b):
					weights_matrix_new[i][j] = u_b
			else:
				weights_matrix_new[i][j] = l_b + r_4*(u_b - l_b)

	#.....................update 6 delayed output weights by harmony search..........................
	r_5 = randint(0,100)/100.00
	r_6 = randint(0,100)/100.00
	r_7 = randint(0,100)/100.00
	r_8 = randint(0,100)/100.00
	eta_array_new = eta_array
	for i in range(26):
		if(r_5<hmcr):
			eta_array_new[i] = eta_array[i]
			if(r_6<par):
				eta_array_new[i] = eta_array_new[i] - (r_7*bw)
			if(eta_array_new[i]<l_b):
				eta_array_new[i] = l_b
			if(eta_array_new[i]>u_b):
				eta_array_new[i] = u_b
		else:
			eta_array_new[i] = l_b + r_8*(u_b - l_b)






	error_array_second = []
	t = 1501
	while t<1520:
		avg = (z[t] + z[t-1] + z[t-2] + z[t-3] + z[t-4] + z[t-7])/6
		inputs = [z[t], z[t-1], z[t-2], z[t-3], z[t-4], z[t-7], avg]
		inputs_cheby = chebyshev(inputs)
		cheby_inputs = [[[] for j in range(cols)] for i in range(rows)]
		

		#....................form input matrix...............................
		k= 0
		while k<21:
			for i in range(rows):
				for j in range(cols):
					cheby_inputs[i][j] = inputs_cheby[k]
					k += 1
		 

		#....................fuzzify weights.................................
		cheby_nodes_sum_array = []
		center_fuzzy_set = []
		width_fuzzy_set = [[[] for j in range(cols)] for i in range(rows)]
		fuzzy_inputs = [[[] for j in range(cols)] for i in range(rows)]
		for i in range(rows):
			cheby_nodes_sum = 0
			for j in range(cols):
				cheby_nodes_sum += cheby_inputs[i][j]
			center_fuzzy_set.append(cheby_nodes_sum/cols)
			
		for i in range(rows):
			for j in range(cols):
				width_fuzzy_set[i][j] = (cheby_inputs[i][j]-center_fuzzy_set[i])/cols

		for i in range(rows):
			for j in range(cols):
				fuzzy_inputs[i][j] = exp(  ((-1)*(cheby_inputs[i][j] - center_fuzzy_set[i])*(cheby_inputs[i][j] - center_fuzzy_set[i]))  /  (2*width_fuzzy_set[i][j]*width_fuzzy_set[i][j]) )


		#...........select the minimum of the top, middle, lower values........
		top = []
		middle = []
		bottom = []
		for i in range(rows):
				for j in range(cols):
					if j==1:
						top.append(fuzzy_inputs[i][j])
					if j==1:
						middle.append(fuzzy_inputs[i][j])
					if j==1:
						bottom.append(fuzzy_inputs[i][j])
		top_val = min(top)
		middle_val = min(middle)
		bottom_val = min(bottom)

		#...............normalization and locally recurrent layer...............
		if hms==1:
			current_firing_strength_1 = top_val/(top_val + middle_val + bottom_val)
			current_firing_strength_2 = middle_val/(top_val + middle_val + bottom_val)
			current_firing_strength_3 = bottom_val/(top_val + middle_val + bottom_val)

			lambda1 = randint(0,100)/100.00
			lambda2 = randint(0,100)/100.00
			lambda3 = randint(0,100)/100.00

			temporal_firing_strength_1 =  (lambda1*current_firing_strength_1) + ( (1-lambda1) * 0)
			temporal_firing_strength_2 =  (lambda2*current_firing_strength_2) + ( (1-lambda2) * 0)
			temporal_firing_strength_3 =  (lambda3*current_firing_strength_3) + ( (1-lambda3) * 0)

		else:
			temporal_firing_strength_1_new =  (lambda1*current_firing_strength_1) + ( (1-lambda1) * temporal_firing_strength_1)
			temporal_firing_strength_2_new =  (lambda2*current_firing_strength_2) + ( (1-lambda2) * temporal_firing_strength_2)
			temporal_firing_strength_3_new =  (lambda3*current_firing_strength_3) + ( (1-lambda3) * temporal_firing_strength_3)

			temporal_firing_strength_1 = temporal_firing_strength_1_new
			temporal_firing_strength_2 = temporal_firing_strength_2_new
			temporal_firing_strength_3 = temporal_firing_strength_3_new

		#..............compute layer 5.............................................
		raw_outputs = []
		i = 0
		while i<3:
			j=0
			out = 0
			while j<21:
				out += weights_matrix_new[i][j] * inputs_cheby[j]
				j+=1
			raw_outputs.append(tanh(out))
			i+=1

		#..............compute layer 6.............................................
		error_array_new = []
		final_outputs = 0
		if hms<=5:
			final_outputs += temporal_firing_strength_1*raw_outputs[0] + temporal_firing_strength_2*raw_outputs[1] + temporal_firing_strength_3*raw_outputs[2] 
			outputs_at_memory_new.append(final_outputs)
		else:
			final_outputs += temporal_firing_strength_1*raw_outputs[0] + temporal_firing_strength_2*raw_outputs[1] + temporal_firing_strength_3*raw_outputs[2] 
			i=1
			while i<6:
				final_outputs += eta_array_new[hms - i] * outputs_at_memory[hms-i]
				i+=1
			outputs_at_memory_new.append(final_outputs)

		expected_second = z[t+1]
		error_new = err(final_outputs, expected)*err(final_outputs, expected)
		error_array_second.append(error_new)	
		t+=1

	for l in range(len(error_array)):
		if (error_array_new[l]<error_array[l]):
			weights_matrix[l] = weights_matrix_new[0][l]


			#..............compute layer 6.............................................
			final_outputs = 0
			if counter<=5:
				final_outputs += temporal_firing_strength_1*raw_outputs[0] + temporal_firing_strength_2*raw_outputs[1] + temporal_firing_strength_3*raw_outputs[2] 
				outputs_at_memory.append(final_outputs)
			else:
				final_outputs += temporal_firing_strength_1*raw_outputs[0] + temporal_firing_strength_2*raw_outputs[1] + temporal_firing_strength_3*raw_outputs[2] 
				i=0
				while i<6:
					final_outputs += eta_array[counter - i] * outputs_at_memory[counter-i]
					i+=1
				outputs_at_memory.append(final_outputs)

			expected = z[t+1]			
			error = err(final_outputs[counter], expected)*err(final_outputs[counter], expected)
			error_array.append(error)	
			






seed(2)
eta_array = initialize_weights(total_iterations)

while training_generations<=20:
	hmcr = 0.9
	par = 0.45
	bw = exp(-6)
	u_b = 0.999
	l_b = 0.001

	outputs_at_memory = []
	outputs_at_memory_new = []
	raw_outputs = []

	error_array = []
	t = 1501
	while t<1520:
		counter = 1
		while counter<=weights_matrix_hms:
			avg = (z[t] + z[t-1] + z[t-2] + z[t-3] + z[t-4] + z[t-7])/6
			inputs = [z[t], z[t-1], z[t-2], z[t-3], z[t-4], z[t-7], avg]
			inputs_cheby = chebyshev(inputs)
			cheby_inputs = [[[] for j in range(cols)] for i in range(rows)]
			

			#....................form input matrix...............................
			k= 0
			while k<21:
				for i in range(rows):
					for j in range(cols):
						cheby_inputs[i][j] = inputs_cheby[k]
						k += 1
			 

			#....................fuzzify weights.................................
			cheby_nodes_sum_array = []
			center_fuzzy_set = []
			width_fuzzy_set = [[[] for j in range(cols)] for i in range(rows)]
			fuzzy_inputs = [[[] for j in range(cols)] for i in range(rows)]
			for i in range(rows):
				cheby_nodes_sum = 0
				for j in range(cols):
					cheby_nodes_sum += cheby_inputs[i][j]
				center_fuzzy_set.append(cheby_nodes_sum/cols)
				
			for i in range(rows):
				for j in range(cols):
					width_fuzzy_set[i][j] = (cheby_inputs[i][j]-center_fuzzy_set[i])/cols

			for i in range(rows):
				for j in range(cols):
					fuzzy_inputs[i][j] = exp(  ((-1)*(cheby_inputs[i][j] - center_fuzzy_set[i])*(cheby_inputs[i][j] - center_fuzzy_set[i]))  /  (2*width_fuzzy_set[i][j]*width_fuzzy_set[i][j]) )


			#...........select the minimum of the top, middle, lower values........
			top = []
			middle = []
			bottom = []
			for i in range(rows):
					for j in range(cols):
						if j==0:
							top.append(fuzzy_inputs[i][j])
						if j==1:
							middle.append(fuzzy_inputs[i][j])
						if j==2:
							bottom.append(fuzzy_inputs[i][j])
			top_val = min(top)
			middle_val = min(middle)
			bottom_val = min(bottom)

			#...............normalization and locally recurrent layer(layer 4)...............
			if counter==1:
				current_firing_strength_1 = top_val/(top_val + middle_val + bottom_val)
				current_firing_strength_2 = middle_val/(top_val + middle_val + bottom_val)
				current_firing_strength_3 = bottom_val/(top_val + middle_val + bottom_val)

				lambda1 = randint(0,100)/100.00
				lambda2 = randint(0,100)/100.00
				lambda3 = randint(0,100)/100.00

				temporal_firing_strength_1 =  (lambda1*current_firing_strength_1) + ( (1-lambda1) * 0)
				temporal_firing_strength_2 =  (lambda2*current_firing_strength_2) + ( (1-lambda2) * 0)
				temporal_firing_strength_3 =  (lambda3*current_firing_strength_3) + ( (1-lambda3) * 0)

			else:
				temporal_firing_strength_1_new =  (lambda1*current_firing_strength_1) + ( (1-lambda1) * temporal_firing_strength_1)
				temporal_firing_strength_2_new =  (lambda2*current_firing_strength_2) + ( (1-lambda2) * temporal_firing_strength_2)
				temporal_firing_strength_3_new =  (lambda3*current_firing_strength_3) + ( (1-lambda3) * temporal_firing_strength_3)

				temporal_firing_strength_1 = temporal_firing_strength_1_new
				temporal_firing_strength_2 = temporal_firing_strength_2_new
				temporal_firing_strength_3 = temporal_firing_strength_3_new

			
			#..............compute layer 5 with 20 weights sets.............................................
			

			counter += 1
		t+=1
	training_generations += 1

print(error_array)



#...............produce the final set of outputs and errors.......................
		final_outputs = 0
		outputs_at_memory = [[[] for j in range(weights_matrix_rows)] for i in range(number_of_initial_training_generations)]
		
		i = 0
		while i<weights_matrix_hms:
			j=0
			while j<weights_matrix_rows:
				final_outputs += temporal_firing_strength_1*raw_outputs[i][j] + temporal_firing_strength_2*raw_outputs[i][j+1] + temporal_firing_strength_3*raw_outputs[i][j+2] 
				j += 3
			outputs_at_memory.append(final_outputs)
			i+=1

temporal_firing_strength_1 =  (lambda1*current_firing_strength_1) + ( (1-lambda1) * temporal_firing_strength_memory[t-1502][0])
temporal_firing_strength_2 =  (lambda2*current_firing_strength_2) + ( (1-lambda2) * temporal_firing_strength_memory[t-1502][1])
temporal_firing_strength_3 =  (lambda3*current_firing_strength_3) + ( (1-lambda3) * temporal_firing_strength_memory[t-1502][2])

#................modify old eta weights by HS algorithm to generate new weights.............................
		for i in range(eta_matrix_hms):
			r_4 = randint(0,100)/100.00
			r_5 = randint(0,100)/100.00
			r_6 = randint(0,100)/100.00
			for j in range(eta_matrix_cols):
				if(r_4<hmcr):
					eta_matrix_new[i][j] = eta_matrix[i][j]
					if(r_5<par):
						eta_matrix_new[i][j] = eta_matrix_new[k][i][j] - (r_6*bw)
					if(eta_matrix_new[i][j]<l_b):
						eta_matrix_new[i][j] = l_b
					if(eta_matrix_new[i][j]>u_b):
						eta_matrix_new[i][j] = u_b
				else:
					eta_matrix_new[i][j] = l_b + random()*(u_b - l_b)


#train dataset for 1 day ahead
t = 1451
inp_trn = 1
msef = []
inp_trn_arr = []
while t<= 1851:
    e_sq = 0
    avg = (z[t] + z[t-1] + z[t-2] + z[t-3] + z[t-4] + z[t-7])/6
    inputs = [z[t], z[t-1], z[t-2], z[t-3], z[t-4], z[t-7], avg]
    weights = weights_matrix[cntr][0]
    outputs = forward_propagate(weights, inputs)
    expected = z[t+1]
    error = err(outputs, expected)
    e_sq += error*error
    msef.append(e_sq/inp_trn)
    inp_trn_arr.append(inp_trn)
    inp_trn2 += 1
    t+=1
#plot MSE during training
plt.plot(inp_trn_arr, msef, 'g', linewidth=2, label='1 Day ahead')
plt.xlabel("Number of generations")
plt.ylabel("MSE")
plt.title("MSE Caculation during training\nFor 1 day ahead")
plt.legend()
plt.show()


# load and prepare data
filename1='djia_close.csv'
dataset1 = load_csv(filename1)
x = np.array(dataset1)
y = x.astype(np.float)
n = len(y)
#normalize inputs
y_min = min(y)
y_max = max(y)
z2 = []
for i in range(n):
	norm = (y[i] - y_min) / (y_max - y_min)
	z2.append(norm)

# load and prepare data
filename2='nasdaq_close.csv'
dataset2 = load_csv(filename2)
x = np.array(dataset2)
y = x.astype(np.float)
n = len(y)
#normalize inputs
y_min = min(y)
y_max = max(y)
z3 = []
for i in range(n):
	norm = (y[i] - y_min) / (y_max - y_min)
	z3.append(norm)

	plt.plot(djia, counter_array, label='DJIA')
plt.plot(nasdaq, counter_array, label='NASDAQ')