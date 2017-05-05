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

		if self.descriptor == "HOG":
			self.l0_neurons = 576
		elif descriptor == "LBP":
			self.l0_neurons = 4096


		self.weights_0 = ng.nguyen(self.l0_neurons, self.l1_neurons)
		self.weights_1 = ng.nguyen(self.l1_neurons, self.l2_neurons)

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

		bias_0 = self.bias
		bias_1 = self.bias

		if self.descriptor == "HOG":
			image = imagelib.getHog(self.path + image_name)
		elif descriptor == "LBP":
			image = imagelib.getLBP(self.path + image_name)
		
		mlp_input = np.reshape(image, np.size(image))
		self.l0_neurons = len(mlp_input)

		weights_0 = ng.nguyen(self.l0_neurons, self.l1_neurons)
		weights_1 = ng.nguyen(self.l1_neurons, self.l2_neurons)

		self.error_file.write("Execucao em {0} \n\n".format(time.strftime("%d/%m/%Y %H:%M")))
		self.write_config_file()

		#feed forward
		layer_0 = mlp_input
		layer_1 = self.activFunction(np.dot(layer_0, weights_0) + bias_0)
		layer_2 = self.activFunction(np.dot(layer_1, weights_1) + bias_1)

		#error layer 2
		expected_output = output.get_output(image_name)
		y_error = expected_output - layer_2
		avg_y_error = np.mean(np.abs(y_error))

		#calculates weights and bias delta
		#print("--- Y ERROR ---")
		#print(y_error)
		#print("--- derivada ---")
		layer_2D = self.derivative(layer_2)
		layer_2MT = np.matrix(layer_2D).T

		layer_1MT = np.matrix(layer_1).T
		#print(layer_1MT)
		y_error = y_error * layer_2MT
		y_delta = self.alpha * layer_1MT.dot(y_error.T)
		bias_0_delta = self.alpha * y_error
		#repass error to hidden layer (layer 1)
		z_error = y_error.T.dot(weights_1.T) #WEIGHTS 1
		#z_error = y_error.T.dot(self.weights_1.T) #WEIGHTS 1
		# layerT1 = np.matrix(layer_1).T
		layer_1MTD = self.derivative(layer_1)
		layer_1MTD = np.matrix(layer_1MTD)
		z_error = z_error * layer_1MTD.T
		z_delta = self.alpha * np.matrix(layer_0).T.dot(z_error)
		bias_1_delta = self.alpha * z_error

		#self.weights_1 += y_delta
		#self.weights_0 += z_delta

		weights_1 += y_delta
		weights_0 += z_delta
		self.weights_0 = weights_0
		self.weights_1 = weights_1

		self.alpha = self.alpha - (self.alpha / self.epochs)
		
		#print(self.weights_0)
		# print (layer_2)		

	def testing(self, image_name):
		mlp_input = None
		image = None

		bias_0 = self.bias
		bias_1 = self.bias

		if self.descriptor == "HOG":
			image = imagelib.getHog(self.path + image_name)
		elif descriptor == "LBP":
			image = imagelib.getLBP(image_name)

		mlp_input = np.reshape(image, np.size(image))
		l0_neurons = len(mlp_input)

		print ("TESTE IMAGE {}".format(output.get_letter(image_name)))
		layer_0 = mlp_input
		layer_1 = self.activFunction(np.dot(layer_0, self.weights_0) + bias_0)
		layer_2 = self.activFunction(np.dot(layer_1, self.weights_1) + bias_1)

		print (layer_2)
		print("\n")


	def run(self, training_data, testing_data):
		self.config_file = open("config.txt", "w")
		self.error_file = open("error.txt", "w")

		random.shuffle(training_data)
		random.shuffle(testing_data)
		print ("Kfold with N epochs started at: {0}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
		for i in range(self.epochs):
			print ("--- EPOCH {0} --- ".format(i))
		# for i in range(2):
			for image in training_data:
				self.training(image)
				
			for image in testing_data:
				self.testing(image)
		print ("Kfold with N epochs ended at: {0}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
		self.config_file.close()
		self.error_file.close()

	