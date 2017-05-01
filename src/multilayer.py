import numpy as np

def activFunction(x):
	return 1/(1+np.exp(-x))	

def derivative(x):
	return x*(1-x)

if __name__ == "__main__":
	bias0 = 1
	bias1 = 1
	alpha = 1.0 #learning rate
	maxEpochs = 100000
	epochs = 0

	imput = np.array([[0,0,1],
	            [0,1,1],
	            [1,0,1],
	            [1,1,1]])
	                
	expected_output = np.array([[0],
				[1],
				[1],
				[0]])

	np.random.seed(1)
	#initialize weights -random
	weights0 = 2*np.random.random((3,4)) - 1
	weights1 = 2*np.random.random((4,1)) - 1


	while epochs < maxEpochs:
		if alpha < 0: 
			break
		#feed forward
		layer0 = imput
		layer1 = activFunction(np.dot(layer0,weights0) + bias0)
		layer2 = activFunction(np.dot(layer1,weights1) + bias1)

		#error layer 2
		y_error = (expected_output - layer2)
		#print ("Error:" + str(np.mean(np.abs(y_error)))) 
		y_error = y_error * derivative(layer2)
		y_delta = alpha * layer1.T.dot(y_error)
		bias0_delta = alpha * y_error

		#repass error to hidden layer
		z_error = y_error.dot(weights1.T)
		z_error = z_error * derivative(layer1)
		z_delta = alpha * layer0.T.dot(z_error)
		bias1_delta = alpha * z_error

		weights1 += y_delta
		weights0 += z_delta
		epochs += 1
		alpha = alpha - 0.00000000000002

	print (layer2)