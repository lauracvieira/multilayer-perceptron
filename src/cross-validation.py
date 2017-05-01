import numpy
import os

def kfold(training):
	num_folds = 5
	subset_size = int(len(training)/num_folds)
	for i in range(num_folds):
		testing_this_round = training[i*subset_size:][:subset_size]      
		training_this_round = training[:i*subset_size] + training[(i+1)*subset_size:]
		# train using training_this_round
		# evaluate against testing_this_round
		# save accuracy
        
    # find mean accuracy over all rounds

if __name__ == "__main__":
	bias = 1
	learningRate = 1
	maxEpochs = 10
	epochs = 0
	#weights = 0 ??

	dataset = os.listdir('./data/dataset1/treinamento')
	print("Quantidade de imagens: {0}".format(len(dataset)))
	kfold(dataset)


	#while the stop condition is false (learning rate, max expochs achieved, error rate)
	while epochs < maxEpochs:
		#feedforward
		#retro propagation
		#weights update
		#test stop condition
		epochs += 1

