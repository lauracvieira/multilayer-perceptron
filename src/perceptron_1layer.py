import numpy as np

if __name__ == "__main__":
	bias = 1
	learningRate = 1
	maxEpochs = 10000
	epochs = 0
	#weights = 0 ??

	# sigmoid function

	def activFunction(x):
		return 1/(1+np.exp(-x))	

	def deriv(x):
		return x*(1-x)

	# input dataset
	#3 inputs
	x = np.array([  [0,0,1],
					[0,1,1],
					[1,0,1],
					[1,1,1] ])

	# output dataset           
	y = np.array([[0,0,1,1]]).T

	# seed random numbers to make calculation
	# deterministic (just a good practice)
	np.random.seed(1)

	# initialize weights randomly with mean 0
	# 3 weights because its 3 imputs
	syn0 = 2*np.random.random((3,1)) - 1

	#while the stop condition is false (learning rate, max expochs achieved, error rate)
	while epochs < maxEpochs:

		#retro propagation
		#weights update
		#test stop condition

		#feedforward
		l0 = x
		y_in = np.dot(l0, syn0) #input multiplied by the the weights
		l1 = activFunction(y_in)

		#calculates error
		l1_error = y - l1
		error = l1_error * deriv(l1)

		#updates weight
		syn0 = syn0 + np.dot(l0.T, error)
		epochs += 1

	print ("Output After Training:")
	print (l1)
