import numpy as np
import nguyen as ng
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
	config_file.write("rede_inicializacao_pesos: nguyen-widrow\n")
	config_file.write("rede_max_epocas: {0}\n".format(maxEpochs))
	config_file.write("rede_tecnica_ajuste_alpha: subtracao de alpha por um valor pequeno\n")

if __name__ == "__main__":
	bias0 = 1
	bias1 = 1
	alpha = 1.0 #learning rate
	maxEpochs = 10000
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
		layer0 = imput
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