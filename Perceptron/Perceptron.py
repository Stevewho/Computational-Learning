
import numpy as np
from numpy import array
import matplotlib.pyplot as plt  

TrainRawData= open("train-a1-449.txt", 'r+')
TestRawData=open("test-a1-449.txt")


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


#function for checking mistake
def mistake_finder(w, dataset):
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
w = pla(dataset)

print("Weight vector\n",w)
# np.savetxt("Weight_Vector.txt", w)


#===testing Data====

#reshape data format as a array
tem=[]
for line in TestRawData:
	line=line.rstrip()
	tem.append(line.split(" "))
	
testdataset= np.array(tem, dtype = np.float).reshape((8,1024))


w=np.array(w).reshape((1,1024))

#mutiply weight and testdata
wx=(w.dot(testdataset.T))


label=[]
for line in wx:
	for x in line:
		label.append(int(np.sign(x)))

print("\nLable of testing data \n",label)

