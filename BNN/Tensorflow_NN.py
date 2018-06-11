
'''
1. nomalize data 
2.  Testing data is for course submittion
'''
import tensorflow as tf
import numpy as np


# normalize https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized_one(a, axis=-1, order=2):
	l2 = np.linalg.norm(a, order, axis)
	l2[l2==0] = 1
	return a / np.expand_dims(l2, axis)



#data import and processing
traindata= np.loadtxt("train-a2-449.txt")
testdata = np.loadtxt("test-a2-449.txt")
y = np.array(traindata[:,-1]).reshape(900,1)
x_test_raw=np.array(testdata).reshape(100,1000)


#normalize data
x=normalized_one(traindata[:,0:1000])
x_test=normalized_one(x_test_raw)


num_neuron=2

#tensorflow variable of weight and bias for 2 layer
w1= tf.Variable(tf.random_uniform([1000,num_neuron], -1.0, 1.0))  # 1000*3 matrix
w0=tf.Variable(tf.random_uniform([num_neuron,1], -1.0, 1.0)) # 3*1 matrix

b1 = tf.Variable(tf.zeros([num_neuron])+0.01)
b0 = tf.Variable(tf.zeros([1])+0.01)

#training example data frame
#https://stackoverflow.com/questions/36693740/whats-the-difference-between-tf-placeholder-and-tf-variable
Xholder= tf.placeholder(tf.float32, shape=(900,1000))
Yholder= tf.placeholder(tf.float32, shape=(900,1))
textholder=tf.placeholder(tf.float32, shape=(100,1000))

#output in each layer
y1 = tf.tanh(tf.matmul(Xholder,w1)+b1)
y0 = tf.tanh(tf.matmul(y1,w0)+b0)

#loss function
loss=tf.reduce_mean(tf.square(y-y0))

#optimizer 
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


#initalize Tensorflow
init=tf.initialize_all_variables()
sess =tf.Session()
sess.run(init)

#training
for step in range(10000):
	sess.run(train, feed_dict={Xholder:x, Yholder:y})
	if step %100 ==0:
		print(step, sess.run(loss, feed_dict={Xholder:x, Yholder:y}))





#inspecting training result

y_hat=sess.run(y0,feed_dict={Xholder:x, Yholder:y})
y_convert=np.sign(y_hat)
compare=np.column_stack((y_convert,y))
# np.savetxt('TFresult.txt', compare, delimiter=' ')

#counting error in training result
n=0
k=0
for i in compare:
	if i[0]==i[1]:
		k+=1
	n+=1
print("correct=",k ,"\ntotal=",n,'\nrate of correct =',k/n)


#testing result:
y1_test=tf.tanh(tf.matmul(textholder,w1)+b1)
y0_test=tf.tanh(tf.matmul(y1_test,w0)+b0)
ytest=sess.run(y0_test, feed_dict={textholder: x_test})
np.savetxt("TestDataOutput.txt", np.sign(ytest))




