import numpy as np
import nguyen as ng
import time
import imagelib #file written to encapsulate the image processing functions

#sigmoid function - activation
def activFunction(x):
	return 1/(1+np.exp(-x))	

def derivative(x):
	return x*(1-x)

#function to write in the config.txt
def writeConfigFile(config_file, alpha, l1_neurons, l2_neurons, maxEpochs):
	config_file.write("Execucao em {0} \n\n".format(time.strftime("%d/%m/%Y %H:%M")))

	config_file.write("extrator: HOG\n")
	config_file.write("extrator_orientacoes: 9\n")
	config_file.write("extrator_pixel_por_celula: 8\n")
	config_file.write("extrator_celula_por_bloco: 1\n\n")
	config_file.write("extrator: LBP \n")
	config_file.write("extrator_orientacoes: \n")
	config_file.write("extrator_pixel_por_celula: \n")
	config_file.write("extrator_celula_por_bloco: \n\n")

	config_file.write("rede_alpha: {0}\n".format(alpha))
	config_file.write("rede_camada_1_neuronios: {0}\n".format(l1_neurons))
	config_file.write("rede_camada_1_funcao_ativacao: sigmoide\n")
	config_file.write("rede_camada_2_neuronios: {0}\n".format(l2_neurons))
	config_file.write("rede_camada_0_funcao_ativacao: sigmoide\n")
	config_file.write("rede_inicializacao_pesos: nguyen-widrow\n")
	config_file.write("rede_max_epocas: {0}\n".format(maxEpochs))
	config_file.write("rede_tecnica_ajuste_alpha: subtracao de alpha por um valor pequeno\n")

#function called by each K-fold cross validation iteration to treat the images data and run the MLP for them

##TO-DO: solucionar e tirar os comentarios dessa função
def run(training_data, testing_data, classes_num, descriptor):
	maxEpochs = 10000
	epochs = 0
	output_matrix = np.matlib.identity(classes_num) #identity matrix - 1s in the diagonal

	#while epochs < maxEpochs: //for N epochs
		#for fold in range(len(training_data)): //for each one of the 4 training folds
			#//go through all the images and do like the following:

	#BEGIN# - put inside FOR Loop
	img_name = "src/test_z.png"
	image = None
	if descriptor == "HOG":
		image = imagelib.getHog(img_name)
	elif descriptor == "LBP":
		image = imagelib.getLBP(img_name) 

	print(image)
	#image_class = getImageClass(img_name) - TO-DO THIS FUNCTION
	#expected output = ver como pega
	#para aquela imagem vai ser a coluna da output_matrix corresponde aquela classe
	#Ex: imagem -> Vê que ela é X
		#pega a coluna 1 da matriz
		#imagem -> vê que ela é Y
		#pega a coluna 2 da matriz

	#hardcode para teste - apagar depois
	expected_output = np.array([[0],
				[0],
				[1]])

	#MLP(image, expected_output, classes_num) - FUNCAO DA REDE EM SI não tá rodando ainda porque o numero de neuronios tá errado (l0_neurons e l1_neurons na funcao MLP)
	#END#

#TO-DO tem que arrumar isso. É como se tivesse rodando a rede N vezes pra uma entrada. E na verdade tem que rodar N vezes (ja definidas na função Run) passando uma imagem por vez. 
#Tem que ver isso porque não sei como arruma
def MLP(mlp_input, expected_output, classes_num):
	bias0 = 1
	bias1 = 1
	alpha = 1.0 #learning rate

	##TO-DO COMO PEGAR ESSES VALORES SEM SER HARDCODED? - Numero de neuronios em cada camada
	#SOLUÇÃO #TO-DO ver quantas colunas a nossa matriz de input tem (a matriz do descriptor no caso), deve ter no numpy, (o numero de colunas da matriz de input é o número de neuronios na camada 0)
	#Pra camada 1: #TO-DO não sei como se estipula quantos neuronios tem que ter nela - procurar
	#Pra camada 2 vai ser o numero de classes que a gente tem no fim das contas - (só pra esclarecer)

	#for the layer 2 (final layer), the num of neurons is the number of classes
	l0_neurons = 6
	l1_neurons = 4
	l2_neurons = classes_num 

	#initialize weights -random
	weights0 = ng.nguyen(l0_neurons, l1_neurons)
	weights1 = ng.nguyen(l1_neurons, l2_neurons)

	config_file = open("config.txt", "w")
	error_file = open("error.txt", "w")
	error_file.write("Execucao em {0} \n\n".format(time.strftime("%d/%m/%Y %H:%M")))
	writeConfigFile(config_file, alpha, l1_neurons, l2_neurons, maxEpochs)

	while epochs < maxEpochs:
		#training

		totalEpochErrors = []
		if alpha < 0:
			config_file.write("rede_criterio_parada: alpha < 0")
			break

		#feed forward
		layer0 = mlp_input
		layer1 = activFunction(np.dot(layer0,weights0) + bias0)
		layer2 = activFunction(np.dot(layer1,weights1) + bias1)

		#error layer 2
		y_error = (expected_output - layer2)
		avg_y_error = np.mean(np.abs(y_error))
		totalEpochErrors.append(avg_y_error) 

		#calculates weights and bias delta
		y_error = y_error * derivative(layer2)
		y_delta = alpha * layer1.T.dot(y_error)
		bias0_delta = alpha * y_error

		#repass error to hidden layer (layer 1)
		z_error = y_error.dot(weights1.T)
		z_error = z_error * derivative(layer1)
		z_delta = alpha * layer0.T.dot(z_error)
		bias1_delta = alpha * z_error

		weights1 += y_delta
		weights0 += z_delta
		alpha = alpha - 0.000001

		#calculates the average error for that epoch and prints to the error file
		averageError = np.mean(totalEpochErrors)
		error_file.write("{0};{1};0\n".format(epochs, averageError))

		epochs += 1

	print (layer2)
	config_file.close()
	error_file.close()



#if __name__ == "__main__":

	