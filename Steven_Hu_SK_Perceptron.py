'''
Course: CS 549
Instructor: Christino Tamon 
Steven(Chia-Hsien) Hu
Home Work 2.5:  Perceptron with Scikit-learn

'''


from sklearn.linear_model import Perceptron 
from sklearn.model_selection import train_test_split
import numpy as np
from time import time

#loading data
TrainRawData= open("train-a1-449.txt", 'r+')
TestRawData=open("test-a1-449.txt")

#reading data and covert it to a list
tem=[]
for line in TrainRawData:
	#remove "/n" at end of line
	line=line.rstrip()  

	#replace Y and N
	line=line.replace('N', '-1')
	line=line.replace('Y', '1')
	tem.append(line.split(" "))
	
#array dataframe
dataset= np.array(tem, dtype = np.float).reshape((792,1025))
X=[]
y=[]
#separate data to X and y
for i in dataset:
	X.append(np.array(i[0:-1]))
	y.append(np.array(int(i[-1])))
X=np.array(X)
y=np.array(y).reshape(792,1)

#setup classifier
my_classifier = Perceptron()

#training
for n in range(20):
	my_classifier.fit(X,y)

#result predict
y_hat=my_classifier.predict(X)
compare=my_classifier.score(X,y)

train_result=np.column_stack((y_hat,y))
print("the correct rate is ",compare," %")





