import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

#loading data
TrainRawData= open("train-a1-449.txt", 'r+') #shape(792,1025)

#reading data and covert it into a list
tem=[]

#reorganize label
for line in TrainRawData:
	#remove "/n" at end of line
	line=line.rstrip()
	line=line.replace('N', '-1')
	line=line.replace('Y', '1')
	tem.append(line.split(" "))



#dataframe setup, reshape( (number ,  -1)): -1 means let program feature got by itself. 
dataset= np.array(tem, dtype = np.float).reshape((len(tem),-1))

train, test=train_test_split(dataset, test_size=0.1, random_state=42)

print('test data has %s rows' %len(test[:,]))

#parameter set up
num_neuron=1
training_inputraw=len(train[:,0])
testing_inputraw=len(test[:,0])
inputfeature=len(train[0,:])-1 # the latest one is the label


def Feature_label(rawdata):
	#separate data to X and y
	X= np.array(rawdata[:,0:-1])  # train_feature(712, 1024) 
	y= np.array(rawdata[:,-1], dtype=int).reshape(-1,1) # train_label(712, 1)
	return X, y

train_feature, train_label = Feature_label(train)
test_feature, test_label = Feature_label(test)

#=====================================
#build model with Tensorflow
#=====================================




#perceptron active function with tf: if x>0 set y0 = 1 otherwise y0=0
#return 0 or 1
def step(x):
    is_greater = tf.greater(x, 0) #retrun 1 or 0 when x>0 or x<0
    as_float = tf.to_float(is_greater)  
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled, 1)


#tensorflow variable, weight and bias
w0=tf.Variable(tf.random_uniform([inputfeature,num_neuron], -1.0, 1.0)) # 3*1 matrix
b0 = tf.Variable(tf.zeros([1])+0.01)


#training example data frame shape =(None, ) means the number of instance you feed with future data in while you run sess 
 # training_feature_holder= tf.placeholder(tf.float32, shape=(training_inputraw,inputfeature))
# training_label_holder= tf.placeholder(tf.float32, shape=(training_inputraw,1))

training_feature_holder= tf.placeholder(tf.float32, shape=(None,inputfeature))
training_label_holder= tf.placeholder(tf.float32, shape=(None,1))

Xholder= tf.placeholder(tf.float32, shape=[training_inputraw,inputfeature])
Yholder= tf.placeholder(tf.float32, shape=[training_inputraw,1])

#output of prediction y0
y0 = step(tf.matmul(Xholder,w0))

#MSE
error=tf.subtract(Yholder,y0)
loss=tf.reduce_mean(tf.square(error))


#update weight
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
	err, _ = sess.run([loss, train], feed_dict={Xholder:train_feature, Yholder:train_label})
	print('epoch:', epoch, 'mse:', err)

score= sess.run(tf.equal(tf.argmax(y0),tf.argmax(train_label)), feed_dict={Xholder:train_feature,Yholder:train_label})


#counting error in training result
y0=sess.run(y0,  feed_dict={Xholder:train_feature,Yholder:train_label})
compare=np.column_stack((y0,train_label))

n=0
k=0
for i in compare:
	if i[0]==i[1]:
		k+=1
	n+=1
print("\ncorrect=",k ,"\ntotal=",n,'\nrate of correct =',k/n)






