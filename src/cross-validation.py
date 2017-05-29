#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from operator import xor
import functions as f
import parameters as p
import perceptron
import sys

def get_descriptor():
    if 'HOG' in sys.argv and 'LBP' in sys.argv:
        print("Apenas um descritor pode ser passado como parâmetro. Escolha 'HOG' ou 'LBP'.")
        exit()

    elif 'HOG' in sys.argv:
        return 'HOG'
    elif 'LBP' in sys.argv:
        return 'LBP'
    else:
        print("O descritor deve ser passado em linha de comando e pode ser apenas 'HOG' ou 'LBP'.")
        exit()

def k_fold(dataset, classes_num, parameters):
    """Validação cruzada utilizando o mé todo k-fold"""
    # definição do número de folds e tamanho do subset 
    # (divisão proporcional entre número de folds e quantidade de arquivos por classe)
    num_folds = 5
    subset_size = int(len(dataset[0]) / num_folds)    

    for fold_i in range(num_folds): 
        testing_this_round = list()
        training_this_round = list()
        
        for dataset_j in range(len(dataset)):
            testing_this_round += dataset[dataset_j][fold_i * subset_size:][:subset_size]      
            training_this_round += dataset[dataset_j][:fold_i * subset_size] + \
                dataset[dataset_j][(fold_i + 1) * subset_size:]

        mlp = perceptron.MLP(classes_num, parameters)
        mlp.run(training_this_round, testing_this_round, fold_i)

# início da execução
if __name__ == "__main__":
    # descritor parâmetros, diretórios, classes e dataset
    descriptor = get_descriptor()
    parameters = p.get_parameters(descriptor)
    f.create_directories(['data', 'src', 'output'])
    dataset = f.get_dataset(f.get_classes())
    k_fold(dataset, len(dataset), parameters)