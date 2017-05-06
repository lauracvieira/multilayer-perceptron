import numpy as np
import nguyen as ng
import time
import random
import imagelib
import output
from datetime import datetime


class MLP(object):
	def __init__(self, hidden_layer_neuron, alpha, classes_num, descriptor, path, epochs):
		self.path = path
		self.bias = 1
		self.alpha = alpha
		self.descriptor = descriptor
		self.l0_neurons = None
		self.l1_neurons = hidden_layer_neuron
		self.l2_neurons = classes_num
		self.epochs = epochs
		self.config_file = None
		self.error_file = None
		self.previous_test_error = 10
		self.previous_weights_0 = None
		self.previous_weights_1 = None

		if self.descriptor == "HOG":
			self.l0_neurons = 576
		elif descriptor == "LBP":
			self.l0_neurons = 4096

		self.weights_0 = ng.nguyen(self.l0_neurons, self.l1_neurons)
		self.weights_1 = ng.nguyen(self.l1_neurons, self.l2_neurons)
		self.avg_test_error = 0
		self.test_number = 0
		self.avg_training_error = 0
		self.training_number = 0


	#sigmoid function - activation
	def activFunction(self, x):
		return 1 / (1 + np.exp(-x))


	def derivative(self, x):
		return x * (1 - x)


	#function to write in the config.txt
	def write_config_file(self):
		self.config_file.write("Execucao em {0} \n\n".format(time.strftime("%d/%m/%Y %H:%M")))
		self.config_file.write("extrator: HOG\n")
		self.config_file.write("extrator_orientacoes: 9\n")
		self.config_file.write("extrator_pixel_por_celula: 8\n")
		self.config_file.write("extrator_celula_por_bloco: 1\n\n")
		self.config_file.write("extrator: LBP \n")
		self.config_file.write("extrator_orientacoes: \n")
		self.config_file.write("extrator_pixel_por_celula: \n")
		self.config_file.write("extrator_celula_por_bloco: \n\n")
		self.config_file.write("rede_alpha: {0}\n".format(self.alpha))
		self.config_file.write("rede_camada_1_neuronios: {0}\n".format(self.l1_neurons))
		self.config_file.write("rede_camada_1_funcao_ativacao: sigmoide\n")
		self.config_file.write("rede_camada_2_neuronios: {0}\n".format(self.l2_neurons))
		self.config_file.write("rede_camada_0_funcao_ativacao: sigmoide\n")
		self.config_file.write("rede_inicializacao_pesos: nguyen-widrow\n")
		self.config_file.write("rede_max_epocas: {0}\n".format(self.epochs))
		self.config_file.write("rede_tecnica_ajuste_alpha: subtracao de alpha por um valor pequeno\n")


	def training(self, image_name):
		mlp_input = None
		image = None

		bias0 = self.bias
		bias1 = self.bias

		if self.descriptor == "HOG":
			image = imagelib.getHog(self.path + image_name)
		elif self.descriptor == "LBP":
			image = imagelib.getLBP(self.path + image_name)
		
		# print(np.size(image))
		mlp_input = np.array(image.reshape(1,np.size(image)))
		self.l0_neurons = len(mlp_input)
		expected_output = np.array(output.get_output(image_name))

		previous_weights_0 = self.weights_0
		previous_weights_1 = self.weights_1

		weights0 = self.weights_0
		weights1 = self.weights_1

		self.write_config_file()


		#feed forward
		layer0 = mlp_input
		layer1 = self.activFunction(np.dot(layer0,weights0) + bias0) #->1x6 (1x576 por 576x6)
		layer2 = self.activFunction(np.dot(layer1,weights1) + bias1).T #->1X3 (1x6 por 6x3)
		y_error = (expected_output - layer2) # 3x1 - 1X3(T) = 1X3
		avg_y_error = np.mean(np.abs(y_error)) #erro médio de uma imagem
		self.avg_training_error = self.avg_training_error + avg_y_error**2
		self.training_number = self.training_number + 1

		#-------------- ATE AQUI TA CERTO ------------------	]
		# print(y_error)
		y_error = y_error*self.derivative(layer2) #y_error = 3x1
		y_error = y_error.T
		y_delta = self.alpha * layer1.T.dot(y_error) #layer1 = 1x1 - y_error = 3x3
		bias0_delta = self.alpha * y_error

		#repass error to hidden layer (layer 1)
		z_error = y_error.dot(weights1.T)
		z_error = z_error * self.derivative(layer1)
		z_delta = self.alpha * layer0.T.dot(z_error)
		bias1_delta = self.alpha * z_error

		weights1 += y_delta
		weights0 += z_delta
		self.weights_1 = weights1
		self.weights_0 = weights0
		
		# print(weights1)
		# print(weights0)
		#print (layer2)		

	def testing(self, image_name):
		mlp_input = None
		image = None
		# totalEpochErrors = []
		bias_0 = self.bias
		bias_1 = self.bias

		if self.descriptor == "HOG":
			image = imagelib.getHog(self.path + image_name)
		elif self.descriptor == "LBP":
			image = imagelib.getLBP(image_name)

		mlp_input = np.reshape(image, np.size(image))
		l0_neurons = len(mlp_input)

		print ("TESTE IMAGE {}".format(output.get_letter(image_name)))
		layer_0 = mlp_input
		layer_1 = self.activFunction(np.dot(layer_0, self.weights_0) + bias_0)
		layer_2 = self.activFunction(np.dot(layer_1, self.weights_1) + bias_1)

		#error layer 2
		expected_output = np.array(output.get_output(image_name))
		y_error = (expected_output - layer_2)
		avg_y_error = np.mean(np.abs(y_error)) #erro médio de uma imagem
		self.avg_test_error = self.avg_test_error + avg_y_error**2
		self.test_number = self.test_number + 1

		print (layer_2)
		# print("Erros totais: {0}".format(totalEpochErrors))
		print("\n")

	def run(self, training_data, testing_data):
		self.config_file = open("outputs/config.txt", "w")
		self.error_file = open("outputs/error.txt", "w")

		random.shuffle(training_data)
		random.shuffle(testing_data)
		self.error_file.write("Execucao em {0} \n\n".format(time.strftime("%d/%m/%Y %H:%M")))
		print ("Kfold with N epochs started at: {0}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
		for i in range(self.epochs):
			print ("--- EPOCH {0} --- ".format(i))
		# for i in range(2):
			for image in training_data:
				self.training(image)
			self.avg_training_error = self.avg_training_error/self.training_number
			#print("Erro quadratico meio dessa epoca de treinamento: {0}".format(self.erro_treino_medio))	

			for image in testing_data:
				self.testing(image)
			self.avg_test_error = self.avg_test_error/self.test_number
			#print("Erro quadratico medio dessa epoca de testes: {0}".format(self.erro_teste_medio))
			self.error_file.write("{0};{1};{2}\n".format(i, self.avg_training_error, self.avg_test_error))

			#if self.avg_test_error > self.previous_test_error:
				#self.weights_0 = self.previous_weights_0
				#self.weights_1 = self.previous_weights_1
				#break
			
			self.alpha = self.alpha - (1 / self.epochs)
			self.previous_test_error = self.avg_test_error
			self.avg_training_error = 0
			self.avg_test_error = 0	
			self.test_number = 0
			self.training_number = 0

		print ("Kfold with N epochs ended at: {0}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
		self.config_file.close()
		self.error_file.close()
