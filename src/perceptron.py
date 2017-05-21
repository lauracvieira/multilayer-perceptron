#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import functions as f
import time
import random
import imagelib
import sys
from datetime import datetime

class MLP(object):
    """Classe que representa a estrutura do multilayer perceptron"""
    def __init__(self, hidden_layer_neuron, alpha, classes_num, descriptor, path,
                 epochs, descriptor_param_1, descriptor_param_2, descriptor_param_3):
        # diretório
        self.path = path
        
        # bias
        self.bias = 1

        # alpha
        self.alpha_initial = alpha
        self.alpha = alpha
        
        # neurônios das camadas
        self.l0_neurons = None
        self.l1_neurons = hidden_layer_neuron
        self.l2_neurons = classes_num
        
        # épocas
        self.epochs = epochs

        # arquivos
        self.config_f = None
        self.error_f = None

        # pesos anteriores
        self.weights_0_previous = None
        self.weights_1_previous = None

        # erros
        self.errors_list = list()
        self.errors_avg = list()

        # descritor
        self.descriptor = descriptor

        # descritores de imagem: parâmetros
        self.descriptor_param_1 = descriptor_param_1
        self.descriptor_param_2 = descriptor_param_2
        self.descriptor_param_3 = descriptor_param_3

        # erros da época 
        self.error_test_avg = 0
        self.test_number = 0
        self.error_training_avg = 0
        self.training_number = 0

        # passagem de imagem de teste para o descritor escolhido 
        # feito para capturar o tamanho da entrada da camada 0 com os parâmetros escolhidos
        # possibilitando a inicializaccão os pesos
        if self.descriptor == "HOG":
            image = imagelib.getHog("./data/img_test.png", self.descriptor_param_1, 
                self.descriptor_param_2, self.descriptor_param_3)
        elif self.descriptor == "LBP":
            image = imagelib.getLBP("./data/img_test.png", self.descriptor_param_1, 
                self.descriptor_param_2)
        self.l0_neurons = np.size(image)

        # pesos: inicialização
        self.weights_0 = f.nguyen(self.l0_neurons, self.l1_neurons)
        self.weights_1 = f.nguyen(self.l1_neurons, self.l2_neurons)


    def activFunction(self, x):
        """Função de ativação"""
        return 1 / (1 + np.exp(-x))


    def derivative(self, x):
        """Derivada"""
        return x * (1 - x)

    def config_write(self):
        """Gravação do arquivo de configuração 'config.txt'"""
        self.config_f.write("Execucao em {0} \n\n".format(time.strftime("%d/%m/%Y %H:%M")))

        if self.descriptor == "HOG":
            self.config_f.write("extrator: HOG\n")
            self.config_f.write("extrator_orientacoes: {}\n".format(self.descriptor_param_1))
            self.config_f.write("extrator_pixel_por_celula: {}\n".format(self.descriptor_param_2))
            self.config_f.write("extrator_celula_por_bloco: {}\n\n".format(self.descriptor_param_3))
        elif self.descriptor == "LBP":
            self.config_f.write("extrator: LBP \n")
            self.config_f.write("extrator_set_points: {}\n".format(self.descriptor_param_1))
            self.config_f.write("extrator_radius: {}\n".format(self.descriptor_param_2))

        self.config_f.write("rede_alpha: {0}\n".format(self.alpha_initial))
        self.config_f.write("rede_camada_1_neuronios: {0}\n".format(self.l1_neurons))
        self.config_f.write("rede_camada_1_funcao_ativacao: sigmoide\n")
        self.config_f.write("rede_camada_2_neuronios: {0}\n".format(self.l2_neurons))
        self.config_f.write("rede_camada_0_funcao_ativacao: sigmoide\n")
        self.config_f.write("rede_inicializacao_pesos: nguyen-widrow\n")
        self.config_f.write("rede_max_epocas: {0}\n".format(self.epochs))
        self.config_f.write("rede_tecnica_ajuste_alpha: alpha - 0.001 para alpha maior que 0\n")
        self.config_f.write("condicao_parada_rede: erro_medio_1 < erro_medio_2 < erro_medio_3 < erro_medio_4 < erro_medio_5\n\n")


    def training(self, image_name, image_i, epoch):
        """Método de treinamento da rede"""
        mlp_input = None
        image = None

        bias_0 = self.bias
        bias_1 = self.bias

        if self.descriptor == "HOG":
            image = imagelib.getHog(self.path + image_name, self.descriptor_param_1, 
                self.descriptor_param_2, self.descriptor_param_3)
        elif self.descriptor == "LBP":
            image = imagelib.getLBP(self.path + image_name, self.descriptor_param_1, 
                self.descriptor_param_2)
        
        # camada de entrada: preparação
        mlp_input = np.array(image.reshape(1, np.size(image)))
        self.l0_neurons = len(mlp_input)
        expected_output = np.array(f.get_output(image_name))

        # pesos: cópia dos pesos antigos
        weights_0_previous = self.weights_0
        weights_1_previous = self.weights_1

        # pesos: preparação dos pesos para esta execução
        weights_0 = self.weights_0
        weights_1 = self.weights_1
        
        # feed forward
        layer_0 = mlp_input
        layer_1 = self.activFunction(np.dot(layer_0,weights_0) + bias_0) #->1x6 (1x576 por 576x6)
        layer_2 = self.activFunction(np.dot(layer_1,weights_1) + bias_1).T #->1X3 (1x6 por 6x3)
        y_error = (expected_output - layer_2) # 3x1 - 1X3(T) = 1X3

        # erro quadrático médio de uma imagem
        avg_y_error = np.sum((y_error) ** 2) / 2 
        self.error_training_avg = self.error_training_avg + avg_y_error
        self.training_number = self.training_number + 1

        # erros: segunda camada
        y_error = y_error * self.derivative(layer_2) #y_error = 3x1
        y_error = y_error.T
        y_delta = self.alpha * layer_1.T.dot(y_error) #layer1 = 1x1 - y_error = 3x3
        bias_0_delta = self.alpha * y_error

        # erros: repasse para a camada escondida
        z_error = y_error.dot(weights_1.T)
        z_error = z_error * self.derivative(layer_1)
        z_delta = self.alpha * layer_0.T.dot(z_error)
        bias_1_delta = self.alpha * z_error

        # pesos: atualização
        weights_1 += y_delta
        weights_0 += z_delta
        self.weights_1 = weights_1
        self.weights_0 = weights_0

        print('Epoch: {0}\tTraining: {1}'.format(epoch, image_i + 1))
        np.savetxt(sys.stdout.buffer, layer_2, '%.10f')
        print("\n")


    def testing(self, image_name, image_i, epoch):
        """Método de teste da rede"""
        mlp_input = None
        image = None
        bias_0 = self.bias
        bias_1 = self.bias

        if self.descriptor == "HOG":
            image = imagelib.getHog(self.path + image_name, self.descriptor_param_1, 
                self.descriptor_param_2, self.descriptor_param_3)
        elif self.descriptor == "LBP":
            image = imagelib.getLBP(self.path + image_name, self.descriptor_param_1, 
                self.descriptor_param_2)

        mlp_input = np.array(image.reshape(1, np.size(image)))
        self.l0_neurons = len(mlp_input)
        expected_output = np.array(f.get_output(image_name))

        print ("Epoch: {0}\tTest: {1}\tImage: {2}".format(epoch, image_i + 1, f.get_letter(image_name)))
        layer_0 = mlp_input
        layer_1 = self.activFunction(np.dot(layer_0, self.weights_0) + bias_0)
        layer_2 = self.activFunction(np.dot(layer_1, self.weights_1) + bias_1).T

        # erros: segunda camada
        y_error = (expected_output - layer_2)

        # erro quadrático médio da imagem        
        avg_y_error = np.sum((y_error) ** 2) / 2
        self.error_test_avg = self.error_test_avg + avg_y_error
        self.test_number = self.test_number + 1

        np.savetxt(sys.stdout.buffer, layer_2, '%.10f')
        print("\n")

    def run(self, training_data, testing_data, fold_num):
        """Método principal de execução do multilayer perceptron"""
        fold_num = fold_num + 1
        self.config_f = open("output/config_{0}.txt".format(fold_num), "w")
        self.error_f = open("output/error_{0}.txt".format(fold_num), "w")
        self.config_write()

        # Torna aleatória a lista de arquivos para treinamento e teste
        random.shuffle(training_data)
        random.shuffle(testing_data)

        self.error_f.write("Execucao em {0} \n\n".format(time.strftime("%d/%m/%Y %H:%M")))
        print ("\nK-Fold with {0} epochs started at: {1}\n".format(self.epochs, 
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        
        for i in range(self.epochs):
            f.print_title_epoch(i, 'training')

            # treinamento de 4/5 do fold
            for image_i, image in enumerate(training_data):
                self.training(image, image_i, i)
            
            # erro médio de treinamento
            self.error_training_avg = self.error_training_avg / self.training_number

            f.print_title_epoch(i, 'testing')
            # teste de 1/5 do fold
            for image_i, image in enumerate(testing_data):
                self.testing(image, image_i, i)

            # erro médio de teste
            self.error_test_avg = self.error_test_avg / self.test_number

             # gravação dos erros quadráticos médios
            self.error_f.write("{0};{1};{2}\n".format(i, self.error_training_avg, self.error_test_avg))
            self.errors_avg.append(self.error_test_avg)

            # serialização dos pesos desta época (model.dat)
            f.serialize_model(fold_num, self.weights_0, self.weights_1)

            # atualização da lista de erros
            f.error_list_update(self.error_test_avg, self.errors_list)

            # atualização da taxa de aprendizado
            if self.alpha - 0.001 > 0:
                self.alpha = self.alpha - 0.001 

            # reinicialização as médias de erros quadráticos com 0 para a próxima época   
            self.error_training_avg = 0
            self.error_test_avg = 0 
            self.test_number = 0
            self.training_number = 0

            if f.stop_condition(self.errors_list) and i > 10:
                break

        # média total
        mean_total = np.mean(self.errors_avg)

        # desvio padrão
        std_dev = np.std(self.errors_avg)

        self.config_f.write("media_total: {0}\n".format(mean_total))
        self.config_f.write("desvio_padrao: {0}\n".format(std_dev))

        print ("\nK-Fold with {0} epochs ended at: {1}\n".format(self.epochs,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        
        self.config_f.close()
        self.error_f.close()