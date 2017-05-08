import os
import perceptron
import argparse
import parameters

def classes(path):
	"""
	Listagem das classes existentes no diretório
	"""
	classes = list()

	for f in os.listdir(path):
		if f[:8] not in classes:
			classes.append(f[:8])

	return classes

def dataset(classes):
	"""
	Reúne os arquivos de cada classe em uma lista de listas
	"""
	dataset = list()

	for i in range(len(classes)):
		files = [f for f in os.listdir(path) if f.startswith(classes[i])]
		dataset.append(files)

	return dataset

def kfold(dataset, hidden_neurons, alpha, classes_num, descriptor, path, epochs, 
		  descriptor_param1, descriptor_param2, descriptor_param3 = 0):
	"""
	Validação cruzada utilizando o mé todo k-fold
	"""

	# Define número de folds e tamanho do subset (que sempre vai ser igualmente dividido entre as classes)
	num_folds = 5
	subset_size = int(len(dataset[0])/num_folds)	

	for fold_i in range(num_folds): 
		testing_this_round = list()
		training_this_round = list()
		
		for dataset_j in range(len(dataset)):
			testing_this_round = testing_this_round + dataset[dataset_j][fold_i*subset_size:][:subset_size]      
			training_this_round = training_this_round + dataset[dataset_j][:fold_i*subset_size] + dataset[dataset_j][(fold_i+1)*subset_size:]
		

		mlp = perceptron.MLP(hidden_neurons, alpha, classes_num, descriptor, path, epochs, descriptor_param1, descriptor_param2, descriptor_param3)
		mlp.run(training_this_round, testing_this_round, fold_i)

def create_directories(directories):
	"""
	Criação dos diretórios
	"""
	for directory in directories:
		try:
		    os.stat('./{0}'.format(directory))
		except:
		    os.mkdir('./{0}'.format(directory))

if __name__ == "__main__":
	create_directories(['data', 'src', 'output'])
	# Definição do diretório de trabalho
	path = './data/dataset1/treinamento/'
	classes = classes(path)
	dataset = dataset(classes)

	if parameters.descriptor in ['HOG', 'LBP']:
		if parameters.descriptor == 'HOG':
			kfold(dataset, parameters.hidden_neurons, parameters.alpha, len(classes), parameters.descriptor, path, parameters.epochs, parameters.hog_orientations, parameters.hog_pixels_per_cell, parameters.hog_cells_per_block)
		elif descriptor == 'LBP':
			kfold(dataset, parameters.hidden_neurons, parameters.alpha, len(classes), parameters.descriptor, path, parameters.epochs, parameters.lbp_points, parameters.lbp_radius)
	else:
		print("O descritor deve ser passado no arquivo de parametros e deve ser'HOG' ou 'LBP'")

