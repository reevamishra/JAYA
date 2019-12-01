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

# Load a CSV file
def load_csv(djia_close):
	dataset = list()
	with open(djia_close, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

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

# load and prepare data
filename='bse500_close.csv'
dataset = load_csv(filename)
x = np.array(dataset)
y = x.astype(np.float)
n = len(y)
#normalize inputs
y_min = min(y)
y_max = max(y)
z1 = []
for i in range(n):
	norm = (y[i] - y_min) / (y_max - y_min)
	z1.append(norm)

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
filename1='nasdaq_close.csv'
dataset1 = load_csv(filename1)
x = np.array(dataset1)
y = x.astype(np.float)
n = len(y)
#normalize inputs
y_min = min(y)
y_max = max(y)
z3 = []
for i in range(n):
	norm = (y[i] - y_min) / (y_max - y_min)
	z3.append(norm)

bse = []
djia = []
nasdaq = []

counter =1 
counter_array = []

for i in range(3000):
	bse.append(z1[i])
	djia.append(z2[i])
	nasdaq.append(z3[i])
	counter_array.append(counter)
	counter+=1
#Original Plot for dataset
plt.plot(counter_array, bse, label='BSE500')
plt.plot(counter_array, djia, label='DJIA')
plt.plot(counter_array, nasdaq, label='NASDAQ')
plt.xlabel("Number of days")
plt.ylabel("Normalised Stockprices")
plt.title("Stock Market Prediction Original Plot for Datasets: \n bse500.csv, djia.csv and nasdaq.csv")
plt.legend()
plt.show()