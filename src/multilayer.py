import numpy as np
import time

def activFunction(x):
	return 1/(1+np.exp(-x))	

def derivative(x):
	return x*(1-x)

def writeConfigFile(config_file, alpha, l1_neurons, l2_neurons, maxEpochs):
	config_file.write("Execucao em {0} \n\n".format(time.strftime("%d/%m/%Y %H:%M")))

	config_file.write("extrator: HOG\n")
	config_file.write("extrator_orientacoes: \n")
	config_file.write("extrator_pixel_por_celula: \n")
	config_file.write("extrator_celula_por_bloco: \n\n")
	config_file.write("extrator: LBP \n")
	config_file.write("extrator_orientacoes: \n")
	config_file.write("extrator_pixel_por_celula: \n")
	config_file.write("extrator_celula_por_bloco: \n\n")

	config_file.write("rede_alpha: {0}\n".format(alpha))
	config_file.write("rede_camada_1_neuronios: {0}\n".format(l1_neurons))
	config_file.write("rede_camada_1_funcao_ativacao: sigmoide\n")
	config_file.write("rede_camada_2_neuronios: {0}\n".format(l2_neurons))
	config_file.write("rede_camada_0_funcao_ativacao: sigmoide\n")
	config_file.write("rede_inicializacao_pesos: aleatoria\n")
	config_file.write("rede_max_epocas: {0}\n".format(maxEpochs))
	config_file.write("rede_tecnica_ajuste_alpha: \n")
	config_file.write("rede_criterio_parada: \n")

if __name__ == "__main__":
	bias0 = 1
	bias1 = 1
	alpha = 1.0 #learning rate
	maxEpochs = 100000
	epochs = 0
	l0_neurons = 3
	l1_neurons = 4
	l2_neurons = 1 

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
	weights0 = 2*np.random.random((l0_neurons,l1_neurons)) - 1
	weights1 = 2*np.random.random((l1_neurons,l2_neurons)) - 1

	config_file = open("config.txt", "w")
	error_file = open("error.txt", "w")
	error_file.write("Execucao em {0} \n\n".format(time.strftime("%d/%m/%Y %H:%M")))
	writeConfigFile(config_file, alpha, l1_neurons, l2_neurons, maxEpochs)

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

		#repass error to hidden layer (layer 1)
		z_error = y_error.dot(weights1.T)
		z_error = z_error * derivative(layer1)
		z_delta = alpha * layer0.T.dot(z_error)
		bias1_delta = alpha * z_error

		weights1 += y_delta
		weights0 += z_delta
		epochs += 1
		alpha = alpha - 0.00000000000002

	print (layer2)
	config_file.close()
	error_file.close()