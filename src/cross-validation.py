#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import utils as u
import parameters as p
import perceptron
import sqlite3
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


def k_fold(dataset, classes_num, parameters, start_algorithm):
    """Validação cruzada utilizando o método k-fold"""
    # definição do número de folds e tamanho do subset
    # (divisão proporcional entre número de folds e quantidade de arquivos por classe)
    num_folds = 5
    subset_size = int(len(dataset[0]) / num_folds)

    # dataset_described = u.de_serialize_dataset(parameters['part_2'])

    conn = sqlite3.connect('./data/database.db')
    cursor = conn.cursor()

    for fold_i in range(num_folds):
        testing_this_round = list()
        training_this_round = list()

        for dataset_j in range(len(dataset)):
            testing_this_round += dataset[dataset_j][fold_i * subset_size:][:subset_size]
            training_this_round += dataset[dataset_j][:fold_i * subset_size] + \
                dataset[dataset_j][(fold_i + 1) * subset_size:]

        mlp = perceptron.MLP(classes_num, parameters, cursor, start_algorithm)
        mlp.run(training_this_round, testing_this_round, fold_i)
    conn.close()


# início da execução
if __name__ == "__main__":
    # horário de execução, descritor, parâmetros, diretórios, lista de classes e lista de dataset
    start_algorithm = datetime.now()
    descriptor = get_descriptor()
    parameters = p.get_parameters(descriptor, 'part2' in sys.argv)
    u.create_directories(['data', 'src', 'output'])
    dataset = u.get_dataset_list(u.get_classes_list(parameters['workpath']), parameters['workpath'])
    k_fold(dataset, len(dataset), parameters, start_algorithm)
    print("Main Start Time:   \t\t\t\t\t\t{}".format(
        start_algorithm.strftime("%Y-%m-%d %H:%M:%S")))
    print("Main End Time:      \t\t\t\t\t\t{}".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("Total time running: \t\t\t\t\t\t{}\n".format(
        datetime.now() - start_algorithm))
