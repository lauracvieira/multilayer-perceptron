#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import perceptron
import parameters as p
import functions as f


def k_fold(dataset, hidden_neurons, alpha, classes_num, descriptor, path, epochs, 
          descriptor_param1, descriptor_param2, descriptor_param3 = 0):
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
        
        mlp = perceptron.MLP(hidden_neurons, alpha, classes_num, descriptor, path, epochs, 
            descriptor_param1, descriptor_param2, descriptor_param3)
        mlp.run(training_this_round, testing_this_round, fold_i)


# início da execução
if __name__ == "__main__":
    # diretórios, classes e dataset
    f.create_directories(['data', 'src', 'output'])
    classes = f.get_classes()
    dataset = f.get_dataset(classes)

    # chamada do k-fold com o descritor definido nos parâmetros
    if p.DESCRIPTOR in ['HOG', 'LBP']:
        if p.DESCRIPTOR == 'HOG':
            k_fold(dataset, p.HIDDEN_NEURONS, p.ALPHA, len(classes), p.DESCRIPTOR, p.WORKPATH,
                p.EPOCHS, p.HOG_ORIENTATIONS, p.HOG_PIXELS_PER_CELL, p.HOG_CELLS_PER_BLOCK)
        elif p.DESCRIPTOR == 'LBP':
            k_fold(dataset, p.HIDDEN_NEURONS, p.ALPHA, len(classes), p.DESCRIPTOR, p.WORKPATH,
                p.EPOCHS, p.LBP_POINTS, p.LBP_RADIUS)
    else:
        print("O descritor deve ser passado no arquivo de parâmetros e pode ser apenas 'HOG' ou 'LBP'")