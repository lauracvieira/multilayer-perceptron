import numpy as np
import funcoes
import time
import random
import imagelib
import sys
from datetime import datetime

class MLP(object):
	"""
	Classe que representa a estrutura do multilayer perceptron
	"""
	def __init__(self, hidden_layer_neuron, alpha, classes_num, descriptor, path,
				 epochs, descriptor_param_1, descriptor_param_2, descriptor_param_3):
		self.path = path
		self.bias = 1
		self.initial_alpha = alpha
		self.alpha = alpha
		self.descriptor = descriptor
		self.l0_neurons = None
		self.l1_neurons = hidden_layer_neuron
		self.l2_neurons = classes_num
		self.epochs = epochs
		self.config_file = None
		self.error_file = None
		# self.previous_test_error = 10
		self.previous_weights_0 = None
		self.previous_weights_1 = None
		self.lista_erros = list()
		self.avg_errors = list()

		#parametros dos descritores de imagem
		self.descriptor_param_1 = descriptor_param_1
		self.descriptor_param_2 = descriptor_param_2
		self.descriptor_param_3 = descriptor_param_3

		#variaveis de calculo de erroƒ"./data/img_test.png"
		self.avg_test_error = 0
		self.test_number = 0
		self.avg_training_error = 0
		self.training_number = 0

		#passagem de uma imagem de teste para o descritor escolhido 
		#feito para capturar o tamanho da entrada da camada 0 com os parametros escolhidos e assim poder inicializar os pesos
		if self.descriptor == "HOG":
			image = imagelib.getHog("./data/img_test.png", self.descriptor_param_1, self.descriptor_param_2, self.descriptor_param_3)
		elif self.descriptor == "LBP":
			image = imagelib.getLBP("./data/img_test.png", self.descriptor_param_1, self.descriptor_param_2)
		self.l0_neurons = np.size(image)

		#inicializacao dos pesos
		self.weights_0 = funcoes.nguyen(self.l0_neurons, self.l1_neurons)
		self.weights_1 = funcoes.nguyen(self.l1_neurons, self.l2_neurons)

	#sigmoid function - activation
	def activFunction(self, x):
		"""
		Função de ativação
		"""
		return 1 / (1 + np.exp(-x))


	def derivative(self, x):
		"""
		Derivada
		"""
		return x * (1 - x)


	#function to write in the config.txt
	def write_config_file(self):
		"""
		Gravação do arquivo de configuração 'config.txt'
		"""
		self.config_file.write("Execucao em {0} \n\n".format(time.strftime("%d/%m/%Y %H:%M")))

		if self.descriptor == "HOG":
			self.config_file.write("extrator: HOG\n")
			self.config_file.write("extrator_orientacoes: {}\n".format(self.descriptor_param_1))
			self.config_file.write("extrator_pixel_por_celula: {}\n".format(self.descriptor_param_2))
			self.config_file.write("extrator_celula_por_bloco: {}\n\n".format(self.descriptor_param_3))
		elif self.descriptor == "LBP":
			self.config_file.write("extrator: LBP \n")
			self.config_file.write("extrator_set_points: {}\n".format(self.descriptor_param_1))
			self.config_file.write("extrator_radius: {}\n".format(self.descriptor_param_2))

		self.config_file.write("rede_alpha: {0}\n".format(self.initial_alpha))
		self.config_file.write("rede_camada_1_neuronios: {0}\n".format(self.l1_neurons))
		self.config_file.write("rede_camada_1_funcao_ativacao: sigmoide\n")
		self.config_file.write("rede_camada_2_neuronios: {0}\n".format(self.l2_neurons))
		self.config_file.write("rede_camada_0_funcao_ativacao: sigmoide\n")
		self.config_file.write("rede_inicializacao_pesos: nguyen-widrow\n")
		self.config_file.write("rede_max_epocas: {0}\n".format(self.epochs))
		self.config_file.write("rede_tecnica_ajuste_alpha: alpha - 0.001 para alpha maior que 0\n")
		self.config_file.write("condicao_parada_rede: erro_medio_1 < erro_medio_2 < erro_medio_3 < erro_medio_4 < erro_medio_5\n\n")


	def training(self, image_name):
		"""
		Método de treinamento da rede 
		"""
		mlp_input = None
		image = None

		bias_0 = self.bias
		bias_1 = self.bias

		if self.descriptor == "HOG":
			image = imagelib.getHog(self.path + image_name, self.descriptor_param_1, self.descriptor_param_2, self.descriptor_param_3)
		elif self.descriptor == "LBP":
			image = imagelib.getLBP(self.path + image_name, self.descriptor_param_1, self.descriptor_param_2)
		
		# prepara as camadas de entrada
		mlp_input = np.array(image.reshape(1, np.size(image)))
		self.l0_neurons = len(mlp_input)
		expected_output = np.array(funcoes.get_output(image_name))

		#guarda os pesos antigos
		previous_weights_0 = self.weights_0
		previous_weights_1 = self.weights_1

		#prepara os pesos que serao utilizados nessa execucao do codigo
		weights_0 = self.weights_0
		weights_1 = self.weights_1
		
		#feed forward
		layer_0 = mlp_input
		layer_1 = self.activFunction(np.dot(layer_0,weights_0) + bias_0) #->1x6 (1x576 por 576x6)
		layer_2 = self.activFunction(np.dot(layer_1,weights_1) + bias_1).T #->1X3 (1x6 por 6x3)
		y_error = (expected_output - layer_2) # 3x1 - 1X3(T) = 1X3
		avg_y_error = np.sum((y_error)**2)/2 #erro quadratico médio de uma imagem
		self.avg_training_error = self.avg_training_error + avg_y_error
		self.training_number = self.training_number + 1

		#error layer 2
		y_error = y_error * self.derivative(layer_2) #y_error = 3x1
		y_error = y_error.T
		y_delta = self.alpha * layer_1.T.dot(y_error) #layer1 = 1x1 - y_error = 3x3
		bias_0_delta = self.alpha * y_error

		#repassa os erros para camada escondida
		z_error = y_error.dot(weights_1.T)
		z_error = z_error * self.derivative(layer_1)
		z_delta = self.alpha * layer_0.T.dot(z_error)
		bias_1_delta = self.alpha * z_error

		#atualizacao dos pesos
		weights_1 += y_delta
		weights_0 += z_delta
		self.weights_1 = weights_1
		self.weights_0 = weights_0

		np.savetxt(sys.stdout.buffer, layer_2, '%.10f')
		print("\n")


	def testing(self, image_name):
		"""
		Método de teste da rede
		"""
		mlp_input = None
		image = None
		bias_0 = self.bias
		bias_1 = self.bias

		if self.descriptor == "HOG":
			image = imagelib.getHog(self.path + image_name, self.descriptor_param_1, self.descriptor_param_2, self.descriptor_param_3)
		elif self.descriptor == "LBP":
			image = imagelib.getLBP(self.path + image_name, self.descriptor_param_1, self.descriptor_param_2)

		mlp_input = np.array(image.reshape(1, np.size(image)))
		self.l0_neurons = len(mlp_input)
		expected_output = np.array(funcoes.get_output(image_name))


		print ("TESTE IMAGE {}".format(funcoes.get_letter(image_name)))
		layer_0 = mlp_input
		layer_1 = self.activFunction(np.dot(layer_0, self.weights_0) + bias_0)
		layer_2 = self.activFunction(np.dot(layer_1, self.weights_1) + bias_1).T

		#error layer 2
		y_error = (expected_output - layer_2)
		avg_y_error = np.sum((y_error)**2)/2 #erro quadratico médio de uma imagem
		self.avg_test_error = self.avg_test_error + avg_y_error
		self.test_number = self.test_number + 1

		np.savetxt(sys.stdout.buffer, layer_2, '%.10f')
		print("\n")

	def run(self, training_data, testing_data, fold_num):
		"""
		Método principal de execução do multilayer perceptron
		"""
		self.config_file = open("output/config{0}.txt".format(fold_num+1), "w")
		self.error_file = open("output/error{0}.txt".format(fold_num+1), "w")
		self.write_config_file()

		random.shuffle(training_data)
		random.shuffle(testing_data)
		self.error_file.write("Execucao em {0} \n\n".format(time.strftime("%d/%m/%Y %H:%M")))
		print ("Kfold with {0} epochs started at: {1}".format(self.epochs,datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
		for i in range(self.epochs):
			print ("--- EPOCH {0} --- ".format(i))
			for image in training_data:
				self.training(image)
			self.avg_training_error = self.avg_training_error/self.training_number

			for image in testing_data:
				self.testing(image)
			self.avg_test_error = self.avg_test_error/self.test_number
			self.error_file.write("{0};{1};{2}\n".format(i, self.avg_training_error, self.avg_test_error)) # Salva os erros quadraticos medios
			self.avg_errors.append(self.avg_test_error)
			funcoes.serialize_model(self.weights_0, self.weights_1)	#Serializacao dos pesos da epoca em questao para o arquivo Model.dat
			funcoes.add_error_list(self.avg_test_error, self.lista_erros)


			if self.alpha - 0.001 > 0:
				self.alpha = self.alpha - 0.001 #atualizacao da taxa de aprendizado

			#zera as medias de erros quadraticos para a proxima epoca	
			self.avg_training_error = 0
			self.avg_test_error = 0	
			self.test_number = 0
			self.training_number = 0

			if funcoes.verify_error(self.lista_erros) and i > 40:
				break

		total_mean = np.mean(self.avg_errors)
		std_dev = np.std(self.avg_errors)
		self.config_file.write("media_total: {0}\n".format(total_mean))
		self.config_file.write("desvio_padrao: {0}\n".format(std_dev))
		print ("Kfold with {0} epochs ended at: {1}".format(self.epochs,datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
		self.config_file.close()
		self.error_file.close()