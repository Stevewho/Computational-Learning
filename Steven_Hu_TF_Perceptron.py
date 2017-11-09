'''
Course: CS 549
Instructor: Christino Tamon 
Steven(Chia-Hsien) Hu
Home Work 2.5:  Perceptron with Tensorflow

'''

import tensorflow as tf
import numpy as np


#website for Perceptron tensorflow
#https://medium.com/@jaschaephraim/elementary-neural-networks-with-tensorflow-c2593ad3d60b
#normalize
#https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy


def normalized(a, axis=-1, order=2):
	l2 = np.linalg.norm(a, order, axis)
	l2[l2==0] = 1
	return a / np.expand_dims(l2, axis)


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
	
#dataframe setup
dataset= np.array(tem, dtype = np.float).reshape((792,1025))

X=[]
y=[]
#separate data to X and y
for i in dataset:
	X.append(np.array(i[0:-1]))
	y.append(np.array(int(i[-1])))
X=np.array(X).reshape(792,1024)
y=np.array(y).reshape(792,1)


#parameter set up
num_neuron=1
inputraw=792
inputfeature=1024



#normalize data
#X=normalized(X)

#perceptron active function with tf: if x>0 set y0 = 1 otherwise y0=0
#return 0 or 1
#https://medium.com/@jaschaephraim/elementary-neural-networks-with-tensorflow-c2593ad3d60b
def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater)
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled, 1)

#tensorflow variable, weight and bias
w0=tf.Variable(tf.random_uniform([1024,num_neuron], -1.0, 1.0)) # 3*1 matrix
b0 = tf.Variable(tf.zeros([1])+0.01)

#training example data frame
#https://stackoverflow.com/questions/36693740/whats-the-difference-between-tf-placeholder-and-tf-variable
Xholder= tf.placeholder(tf.float32, shape=(inputraw,inputfeature))
Yholder= tf.placeholder(tf.float32, shape=(inputraw,1))
textholder=tf.placeholder(tf.float32, shape=(100,1000))


#output of prediction y0
y0 = step(tf.matmul(Xholder,w0))

#MSE
error=tf.subtract(Yholder,y0)

loss=tf.reduce_mean(tf.square(error))


#update weight
'''
Note:
In the website the error term don't divert by 2, it is not match w = w - yx.
Insteat it is w = w - 2yx
'''
delta = tf.matmul(Xholder, error/2, transpose_a=True)
train = tf.assign(w0, tf.add(w0,delta))


#inital setting
init=tf.global_variables_initializer()
sess =tf.Session()
sess.run(init)
#print(sess.run(error, feed_dict={Xholder:X, Yholder:y}))


#training data with TF
err, target = 1, 0
epoch, max_epochs = 0, 20
while err > target and epoch < max_epochs:
	epoch += 1
	err, _ = sess.run([loss, train], feed_dict={Xholder:X, Yholder:y})
	print('epoch:', epoch, 'mse:', err)

score= sess.run(tf.equal(tf.argmax(y0),tf.argmax(y)), feed_dict={Xholder:X,Yholder:y})


#counting error in training result
y0=sess.run(y0,  feed_dict={Xholder:X,Yholder:y})
compare=np.column_stack((y0,y))
n=0
k=0
for i in compare:
	if i[0]==i[1]:
		k+=1
	n+=1
print("\ncorrect=",k ,"\ntotal=",n,'\nrate of correct =',k/n)






