#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import utils as u
import imagelib
import numpy as np
import random
import pandas as pd
import parameters as p
import sqlite3
import sys
import time

class MLP(object):
    """Classe que representa a estrutura do multilayer perceptron"""
    def __init__(self, classes_num, parameters, cursor, start_algorithm):
        # 2ª entrega
        self.part_2 = parameters['part_2']

        # tempo de execução
        self.start_algorithm = start_algorithm 
        self.start_fold = None
        self.end_fold = None
        
        # diretório
        self.path = parameters['workpath']
        
        # bias
        self.bias_0 = 1
        self.bias_1 = 1

        # alpha
        self.alpha = parameters['alpha']
        
        # neurônios das camadas
        self.l0_neurons = None
        self.l1_neurons = parameters['hidden_neurons']
        self.l2_neurons = classes_num
        
        # épocas
        self.epochs = parameters['epochs']

        # arquivos
        self.config_f = None
        self.error_f = None

        #self.
        # pesos anteriores
        self.weights_0_previous = None
        self.weights_1_previous = None

        # lista de erros para a condicao de parada
        self.errors_list = list()
        self.errors_test_avg_list = list()

        # descritor
        self.descriptor = parameters['descriptor']

        # descritores de imagem: parâmetros
        self.descriptor_param_1 = parameters['descriptor_param_1']
        self.descriptor_param_2 = parameters['descriptor_param_2']
        self.descriptor_param_3 = parameters['descriptor_param_3']

        # erros da época 
        self.error_test_avg = 0
        self.test_number = 0
        self.error_training_avg = 0
        self.training_number = 0

        #listas com resultados esperados e obtidos para a matriz de confusao
        self.test_predicted = []
        self.test_results = []
        
        # abertura do dataset serializado e leitura da matriz de descrição da imagem de teste 
        # utilizada para capturar o tamanho da entrada da camada 0 com os parâmetros escolhidos
        # possibilitando a inicializaccão os pesos
        self.cursor = cursor
        try:
            self.cursor.execute('SELECT {} FROM treinamento WHERE image_name = "{}"'.format(self.descriptor, 'img_test.png'))
            row = self.cursor.fetchone()
            image = np.frombuffer(row[0])
        except TypeError:
            print('Erro na obtenção da matriz {} da imagem {} da tabela "treinamento"'.format(self.descriptor, 'img_test.png'))
            print('Execução encerrada')
            exit()

        self.l0_neurons = np.size(image)

        # pesos: inicialização
        self.weights_0 = u.nguyen(self.l0_neurons, self.l1_neurons)
        self.weights_1 = u.nguyen(self.l1_neurons, self.l2_neurons)


    def activFunction(self, x):
        """Função de ativação"""
        return 1 / (1 + np.exp(-x))


    def derivative(self, x):
        """Derivada"""
        return x * (1 - x)

    def config_write(self):
        """Gravação do arquivo de configuração 'config.txt'"""
        self.config_f.write("Execucao em {} \n\n".format(time.strftime("%d/%m/%Y %H:%M")))

        if self.descriptor == "HOG":
            self.config_f.write("extrator: HOG\n")
            self.config_f.write("extrator_orientacoes: {}\n".format(self.descriptor_param_1))
            self.config_f.write("extrator_pixel_por_celula: {}\n".format(self.descriptor_param_2))
            self.config_f.write("extrator_celula_por_bloco: {}\n\n".format(self.descriptor_param_3))
        elif self.descriptor == "LBP":
            self.config_f.write("extrator: LBP \n")
            self.config_f.write("extrator_set_points: {}\n".format(self.descriptor_param_1))
            self.config_f.write("extrator_radius: {}\n".format(self.descriptor_param_2))

        self.config_f.write("rede_alpha: {}\n".format(self.alpha))
        self.config_f.write("rede_camada_1_neuronios: {}\n".format(self.l1_neurons))
        self.config_f.write("rede_camada_1_funcao_ativacao: sigmoide\n")
        self.config_f.write("rede_camada_2_neuronios: {}\n".format(self.l2_neurons))
        self.config_f.write("rede_camada_0_funcao_ativacao: sigmoide\n")
        self.config_f.write("rede_inicializacao_pesos: nguyen-widrow\n")
        self.config_f.write("rede_max_epocas: {}\n".format(self.epochs))
        self.config_f.write("rede_tecnica_ajuste_alpha: alpha - 0.001 para alpha maior que 0\n")
        

    def training(self, image_name, image_i, epoch, fold_num):
        """Método de treinamento da rede"""
        mlp_input = None
        image = None

        bias_0 = self.bias_0
        bias_1 = self.bias_1

        try:
            self.cursor. execute('SELECT {} FROM treinamento WHERE image_name = "{}"'.format(self.descriptor, image_name))
            row = self.cursor.fetchone()
            image = np.frombuffer(row[0])
        except TypeError:
            print('Erro na obtenção da matriz {} da imagem {} da tabela "treinamento"'.format(self.descriptor, image_name))
            print('Execução encerrada')
            exit()

        # camada de entrada: preparação
        mlp_input = np.array(image.reshape(1, np.size(image)))
        self.l0_neurons = len(mlp_input)
        expected_output = np.array(u.get_output(image_name, self.part_2))

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

        # pesos e bias: atualização
        weights_1 += y_delta
        weights_0 += z_delta
        bias_1 += bias_0_delta
        bias_0 += bias_1_delta
        self.weights_1 = weights_1
        self.weights_0 = weights_0
        self.bias_0 = bias_0
        self.bias_1 = bias_1

        print('Fold: {}\tEpoch: {}\tTraining: {}'.format(fold_num, str(epoch).zfill(4),
           str(image_i + 1).zfill(4)))
        np.savetxt(sys.stdout.buffer, layer_2, '%.10f')
        print("\n")
    

    def testing(self, image_name, image_i, dataset_tests = False):
        """Método de teste da rede"""
        mlp_input = None
        image = None
        bias_0 = self.bias_0
        bias_1 = self.bias_1

        table = 'treinamento'
        if dataset_tests == True:
            table = 'testes'

        try:
            self.cursor. execute('SELECT {} FROM {} WHERE image_name = "{}"'.format(self.descriptor, table, image_name))
            row = self.cursor.fetchone()
            image = np.frombuffer(row[0])
        except TypeError:
            print('Erro na obtenção da matriz {} da imagem {} da tabela "testes"'.format(self.descriptor, image_name))
            print('Execução encerrada')
            exit()

        mlp_input = np.array(image.reshape(1, np.size(image)))
        self.l0_neurons = len(mlp_input)
        expected_output = np.array(u.get_output(image_name, self.part_2))

        print ("Test: {}\tImage: {}".format(str(image_i + 1).zfill(4), u.get_letter(image_name, self.part_2)))
        layer_0 = mlp_input
        layer_1 = self.activFunction(np.dot(layer_0, self.weights_0) + bias_0)
        layer_2 = self.activFunction(np.dot(layer_1, self.weights_1) + bias_1).T

        #se o teste estiver sendo rodado para o dataset de testes, armazena os acertos e erros para a matriz de confusao
        if dataset_tests == True:
            resulting_letter = u.get_resulting_letter(layer_2, self.part_2)
            if resulting_letter != None:
                self.test_predicted.append(u.get_letter(image_name, self.part_2))
                self.test_results.append(resulting_letter)

        # erros: segunda camada
        y_error = (expected_output - layer_2)

        # erro quadrático médio da imagem        
        avg_y_error = np.sum((y_error) ** 2) / 2
        self.error_test_avg = self.error_test_avg + avg_y_error
        self.test_number = self.test_number + 1

        np.savetxt(sys.stdout.buffer, layer_2, '%.10f')
        print("\n")

    def confusion_matrix(self):
        pass
        #todo have fun, Laura!

    def run(self, training_data, testing_data, fold_num):
        """Método principal de execução do multilayer perceptron"""
        fold_num = fold_num + 1
        self.config_f = open("output/config_{}_{}_{}.txt".format(fold_num, 
            self.start_algorithm.strftime("%d%m%Y-%H%M"), self.descriptor), "w")
        self.error_f = open("output/error_{}_{}_{}.txt".format(fold_num, 
            self.start_algorithm.strftime("%d%m%Y-%H%M"), self.descriptor), "w")
        self.config_write()

        # Torna aleatória a lista de arquivos para treinamento e teste
        random.shuffle(training_data)
        random.shuffle(testing_data)

        self.start_fold = datetime.now()

        self.error_f.write("Execucao em {} \n\n".format(time.strftime("%d/%m/%Y %H:%M")))
        print ("\nK-Fold with max {} epochs started at: {}\n".format(self.epochs, 
            self.start_fold.strftime("%Y-%m-%d %H:%M:%S")))

        for epoch_current in range(self.epochs):
            u.print_title_epoch(epoch_current + 1, fold_num, 'training', self.part_2, self.descriptor)

            # treinamento de 4/5 do fold
            for image_i, image in enumerate(training_data):
                self.training(image, image_i, epoch_current + 1, fold_num)
            
            # erro médio de treinamento
            self.error_training_avg = self.error_training_avg / self.training_number
  
            # serialização dos pesos desta época (model.dat)
            u.serialize_model(fold_num, self.weights_0, self.weights_1, self.start_algorithm, self.descriptor)

            # teste de 1/5 do fold
            u.print_title_epoch(epoch_current + 1, fold_num,'testing', self.part_2, self.descriptor)        
            for image_i, image in enumerate(testing_data):
                self.testing(image, image_i)

            # erro médio de teste
            self.error_test_avg = self.error_test_avg / self.test_number

            # salva o erro quadratico médio desta época
            self.errors_test_avg_list.append(self.error_test_avg)

            # atualização da lista de erros
            u.error_list_update(self.error_test_avg, self.errors_list)
            
            # gravação dos erros quadráticos médios
            self.error_f.write("{};{};{}\n".format(epoch_current, self.error_training_avg,
             self.error_test_avg))

            # reinicialização das médias de erros quadráticos com 0 para a próxima época   
            self.error_training_avg = 0
            self.error_test_avg = 0 
            self.test_number = 0
            self.training_number = 0

            # atualização da taxa de aprendizado e condição de parada por taxa de aprendizado
            if self.alpha - 0.001 > 0:
                self.alpha = self.alpha - 0.001

            # condicao de parada por erro
            stop_condition = u.stop_condition(self.errors_list, epoch_current, self.alpha)
            if stop_condition['result']:
                break

        #obtencao das imagens da pasta testes daquele dataset para validacao
        #funcao teste chamada para cada imagem, para adicionar os acertos na matriz de confusao posteriormente
        parameters_test = p.get_parameters(self.descriptor, self.part_2)
        dataset_validation = u.get_dataset_list(u.get_classes_list(parameters_test['testpath']), parameters_test['testpath'])
        for dataset in dataset_validation:
            for image_i, image in enumerate(dataset):
                self.testing(image, image_i, dataset_tests = True)

        # matriz de confusão
        obtained = pd.Series(self.test_results, name='Esperado')
        predicted = pd.Series(self.test_predicted, name='   Obtido')
        confusion_matrix = pd.crosstab(obtained, predicted)
        print(confusion_matrix)
        
        # média total
        mean_total = np.mean(self.errors_test_avg_list)

        # desvio padrão
        std_dev = np.std(self.errors_test_avg_list)

        self.config_f.write("epoca_final: {}\n".format(stop_condition['message']))
        self.config_f.write("epoca_final: {}\n".format(epoch_current))
        self.config_f.write("media_total: {}\n".format(mean_total))
        self.config_f.write("desvio_padrao: {}\n".format(std_dev))

        self.end_fold = datetime.now()

        print ("\nK-Fold {}/5:\tMax Epoch (s):  \t{}\tStart Time:\t{}".format(fold_num, self.epochs, 
            self.start_fold.strftime("%Y-%m-%d %H:%M:%S")))

        print ("K-Fold {}/5:\tTotal Epoch (s):\t{}\tEnd Time:\t{}".format(fold_num, epoch_current + 1,
            self.end_fold.strftime("%Y-%m-%d %H:%M:%S")))

        print ("K-Fold {}/5:\t\t\t\t\tRun. Time:\t{}\n".format(fold_num, self.end_fold - self.start_fold))
        
        self.config_f.close()
        self.error_f.close()

        #self.plot_graph()