import os
import perceptron
import argparse

def classes(path):
	"""
	Listagem das classes existentes no diretório
	:Parameters:
		path: diretório de trabalho
	"""
	classes = list()

	for f in os.listdir(path):
		if f[:8] not in classes:
			classes.append(f[:8])

	# ####################################################################### TESTES classes INÍCIO
	# # Teste para imprimir as classes encontradas no diretório de trabalho
	# print("\nClasses encontradas: {0}\n{1}".format(len(classes), classes))
	# ####################################################################### TESTES classes FIM

	return classes

def dataset(classes):
	"""
	Reúne os arquivos de cada classe em uma lista de listas
	:Parameters:
		classes: cada uma das classes a ser buscada
	"""
	dataset = list()

	for i in range(len(classes)):
		files = [f for f in os.listdir(path) if f.startswith(classes[i])]
		dataset.append(files)

	# ####################################################################### TESTES dataset INÍCIO
	# # Teste para imprimir a quantidade de arquivos divididos por classes
	# print("\nQuantidade de classes: {0}".format(len(dataset)))
	# for i in range(len(dataset)):
	# 	print("Classe '{0}': {1} elementos".format(classes[i], len(dataset[i])))
	# ####################################################################### TESTES dataset FIM

	return dataset

def kfold(dataset, hidden_neurons, alpha, classes_num, descriptor, path, epochs):
	"""
	Validação cruzada utilizando o mé todo k-fold
	:Parameters:
		training: lista de imagens contidas em classes (cada classe é uma lista)
	"""

	# Define número de folds e tamanho do subset (que sempre vai ser igualmente dividido entre as classes)
	num_folds = 5
	subset_size = int(len(dataset[0])/num_folds)	

	#for fold_i in range(num_folds):
	for fold_i in range(num_folds): ##TO-DO - tirar essa linha, coloquei só pra não rodar varias vezes
		testing_this_round = list()
		training_this_round = list()
		
		for dataset_j in range(len(dataset)):
			testing_this_round = testing_this_round + dataset[dataset_j][fold_i*subset_size:][:subset_size]      
			training_this_round = training_this_round + dataset[dataset_j][:fold_i*subset_size] + dataset[dataset_j][(fold_i+1)*subset_size:]
		

		mlp = perceptron.MLP(hidden_neurons, alpha, classes_num, descriptor, path, epochs)
		mlp.run(training_this_round, testing_this_round)
		# break;
		
		# ####################################################################### TESTES kfold INÍCIO
		# # Teste para imprimir a quantidade de imagens enviadas para teste e treinamento
		# print("\nk-fold {0}:".format(fold_i+1))
		# print("Quantidade de imagens enviadas para teste: {0}".format(len(testing_this_round)))
		# print("Quantidade de imagens enviadas para treinamento: {0}\n".format(len(training_this_round)))

		# # Teste para imprimir todas as imagens de cada classe
		# last = testing_this_round[0]
		# i = 0
		# for x in testing_this_round:
		# 	if x[:8] not in last:
		# 		print("\n-----------------------------------------------------\n")
		# 		i = 0

		# 	last = x
		# 	i = i + 1
		# 	print("{0}. {1}".format(i, x))
		# ####################################################################### TESTES kfold FIM


    # todo acurácia média em todas as rodadas

if __name__ == "__main__":	
	# Definição do diretório de trabalho
	path = './data/dataset1/treinamento/'
	classes = classes(path)
	dataset = dataset(classes)

	parser = argparse.ArgumentParser(description='Choice of which image descriptor will be used in the MLP') 
	parser.add_argument('--descriptor', '--desc', required= True, help="Choose descriptor HOG or LBP to use in the MLP")
	args = parser.parse_args()
	descriptor = args.descriptor

	epochs = 1
	hidden_neurons = 32
	alpha = 1.0

	if descriptor in ['HOG', 'LBP']:
		kfold(dataset, hidden_neurons, alpha, len(classes), descriptor, path, epochs)
	else:
		print("The descriptor should be 'HOG' or 'LBP'")