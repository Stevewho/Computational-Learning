

'''
Course: CS 549
Home Work 1: Perceptron
'''

import numpy as np
from numpy import array
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split

TrainRawData= open("data-a1-449.txt", 'r+')


tem=[]

#reading data and covert it to a list
for line in TrainRawData:

	line=line.rstrip()  #remove "/n" at end of line

	#replace Y and N
	line=line.replace('N', '-1')
	line=line.replace('Y', '1')
	
	tem.append(line.split(" "))
	

#array dataframe
dataset= np.array(tem, dtype = np.float).reshape((792,1025))
train, test=train_test_split(dataset, test_size=0.1, random_state=42)
print(test.shape)

print('test data has %s rows' %len(test[:,]))

#function for checking mistake
def mistake_finder(w, train):
	result = None
	for i in dataset:
		y=int(i[-1])
		x=np.array(i[0:-1])
		if int(np.sign(w.T.dot(x))) !=y:
			result= x,y
			return result
	return result

#function for running perceptron
def pla(dataset):
	w = np.zeros(1024)
	while mistake_finder(w, dataset) is not None:
		x, y = mistake_finder(w, dataset)
		w += y * x
	return w

#get the wight
w = pla(train)

print("Weight vector\n",w)
# np.savetxt("Weight_Vector.txt", w)


#===testing Data====

#reshape data format as a array
test_data=test[:,0:1024]
test_label=test[:,-1]

w=np.array(w).reshape((1,1024))

#mutiply weight and testdata
wx=(w.dot(test_data.T))

label=[]
for line in wx:
	for x in line:
		label.append(int(np.sign(x)))

output = np.array(label)
error = sum(test_label - output)
print("error rate is %s" %(error/len(test[:,])))
print("\nLable of testing data \n",label)



