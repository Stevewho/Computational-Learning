'''
Course: CS 549
Instructor: Christino Tamon 
Steven(Chia-Hsien) Hu
Assigment 3: Kernal Perceptron


bias =1 
dimension =2 
kernal function = kernal=np.power((np.dot(TrainData[:,0:2],TrainData[j,0:2].T)+bias),2)
'''


import numpy as np
from numpy import array
import matplotlib.pyplot as plt 


TrainData= np.loadtxt("train-data-a3-449.txt")  #(90,2)
label = np.loadtxt('train-label-a3-449.txt' ).reshape(90,1)
print(TrainData.shape)


def mistakefinder(TrainData, label, alpha, bias_k):

	alpha_list=[]
	bias_k_list=[]
	bias_y_list=[]
	result = None
	for j in range(TrainData.shape[0]):
		y=label[j,0]
		kernal=np.power((np.dot(TrainData[:,0:2],TrainData[j,0:2].T)+bias_k),2)
		y_hat=np.sign(np.sum(alpha*y*kernal+bias_y))
		kernal_list.append(kernal)
		if y!=y_hat:
			alpha_add=1
			bias_k_add=1
			bias_y_add=1
			result = alpha_add, bias_add
			return result
	return result


#training
n=0
alpha=0
bias_k =0
bias_y=0

alpha_list=[]
bias_k_list=[]
bias_y_list=[]
y_hat_list=[]
for j in range(TrainData.shape[0]):
		print("sample: ", j)
		y=label[j,0]
		kernal=np.power((np.dot(TrainData[:,0:2],TrainData[j,0:2].T)+bias_k),2)
		y_hat=np.sign(np.sum(alpha*y*kernal+bias_y))

		
		while y!=y_hat:

			kernal=np.power((np.dot(TrainData[:,0:2],TrainData[j,0:2].T)+bias_k),2)
			y_hat=np.sign(np.sum(alpha*y*kernal+bias_y))
		
		


			if y!=y_hat:
				alpha+=1
				bias_k+=1
				bias_y+=1
		print(y, y_hat)
		alpha_list.append(alpha)
		bias_k_list.append(bias_k)
		bias_y_list.append(bias_y)
		y_hat_list.append(y_hat)
		

print('alpha: ',alpha_list)
print('bias_k_list: ',bias_k_list)
print('bias_y_list: ',bias_y_list)

# while mistakefinder(TrainData, label, alpha, bias_k, bias_y) is not None:
# 	alpha_list=[]
# 	bias_k_list=[]
# 	bias_y_list=[]
# 	alpha_add, bias_k_add, bias_y_add  =  mistakefinder(TrainData, label, alpha, bias_k, bias_y)
# 	alpha+= alpha_add
# 	bias_k+= bias_k_add
# 	bias_y += bias_y_add
# 	print('alpha: ',alpha)
# 	alpha_list.append(alpha)
# 	bias_list.append(bias)




#result of training
# y_hat_save=[]
# for j in range(TrainData.shape[0]):
# 	y=label[j,0]
# 	kernal=np.power((np.dot(TrainData[:,0:2],TrainData[j,0:2].T)+bias),2)
# 	# print(kernal)
# 	y_hat=np.sign(np.sum(alpha*y*kernal))
# 	y_hat_save.append(y_hat)
# 	kernal_list.append(kernal)
# print('alpha_list:',alpha_list)

#accuracy calculate
# compare= np.column_stack((np.array(y_hat_save).reshape(90,1),label[:,-1]))
# n=0
# k=0
# for i in compare:
# 	if i[0]==i[1]:
# 		k+=1
# 	n+=1
# print("correct=",k ,"\ntotal=",n,'\nrate of correct =',k/n)

