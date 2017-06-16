#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import parameters as p


def create_directories(directories):
    """Criação dos diretórios"""
    for directory in directories:
        try:
            os.stat('./' + directory)
        except OSError:
            os.mkdir('./' + directory)


def get_classes_list(path):
    """Listagem das classes existentes no diretório"""
    classes = list()

    for f in os.listdir(path):
        if f[:8] not in classes:
            classes.append(f[:8])

    return classes


def get_dataset_list(classes, path):
    """Reúne os arquivos de cada classe em uma lista de listas"""
    dataset = list()

    for i in range(len(classes)):
        files = [f for f in os.listdir(path) if f.startswith(classes[i])]
        dataset.append(files)

    return dataset


def get_classes_dict(part_2):
    """Retorna um dicionário contendo as letras das classes
        correspondentes ao nome físico dos arquivos"""
    if part_2:
        classes = {
            'train_41': 'A',
            'train_42': 'B',
            'train_43': 'C',
            'train_44': 'D',
            'train_45': 'E',
            'train_46': 'F',
            'train_47': 'G',
            'train_48': 'H',
            'train_49': 'I',
            'train_4a': 'J',
            'train_4b': 'K',
            'train_4c': 'L',
            'train_4d': 'M',
            'train_4e': 'N',
            'train_4f': 'O',
            'train_50': 'P',
            'train_51': 'Q',
            'train_52': 'R',
            'train_53': 'S',
            'train_54': 'T',
            'train_55': 'U',
            'train_56': 'V',
            'train_57': 'W',
            'train_58': 'X',
            'train_59': 'Y',
            'train_5a': 'Z'
        }
    else:
        classes = {
            'train_53': 'S',
            'train_58': 'X',
            'train_5a': 'Z'
        }
    return classes


def get_classes_letters_list(part_2):
    """Retorna lista contendo as letras das classes correspondentes à entrega"""
    return list(get_classes_dict(part_2).values())


def get_letter(image_name, part_2):
    """Retorna a letra relacionada à classe do arquivo"""
    return get_classes_dict(part_2)[image_name[:8]]


def get_output(image_name, part_2):
    """Retorna um dicionário contendo a configuração de saída esperada"""
    letras = {}

    for i, letter in (enumerate(get_classes_dict(part_2).values())):
            letras[letter] = i

    letra = get_letter(image_name, part_2)

    if part_2:
        output_matrix = np.matlib.identity(26)
    else:
        output_matrix = np.matlib.identity(3)

    valor = letras[letra]
    linha = output_matrix[valor]

    return linha.T


def serialize_model(fold_num, weight_0, weight_1, start_algorithm,
descriptor, part_2, l1_neurons):
    """Serialiização dos pesos no arquivo output/model.dat"""
    file_command = 'output/model-{desc}-N{hn:03}-P{part}-F{fold}-{datetime}.txt'.format(fold=fold_num,
        datetime=start_algorithm.strftime('%Y-%m-%d-%H-%M-%S.%f'),
            desc=descriptor, part=2 if part_2 else 1, hn=l1_neurons)

    with open(file_command, 'wb') as f:
        pickle.dump((weight_0, weight_1), f)


def initnw(inputs, outputs):
    """Inicialização dos pesos do Nguyen-Widrow"""
    ci = inputs
    cn = outputs
    w_fix = 0.7 * cn ** (1. / ci)
    w_rand = np.random.rand(cn, ci) * 2 - 1
    # Normalize
    if ci == 1:
        w_rand /= np.abs(w_rand)
    else:
        w_rand *= np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(cn, 1))

    w = w_fix * w_rand

    return w


def nguyen(inputs, outputs):
    """Retorno transposto dos pesos do Nguyen-Widrow"""
    neww = []
    neww = initnw(inputs, outputs)

    return neww.T


def stop_condition(error_list, epoch_current, alpha):
    """Aciona parada quando o erro for menor que o erro mínimo
        ou se o erro for crescente por 4 épocas consecutivas E tiver executado o número
        mínimo de épocas estipulado nos parâmetros"""
    condition = {'result': 0}

    if len(error_list) < 4:
        return condition

    if error_list[-1] < p.get_error_min():
        condition['result'] = 1
        condition['message'] = 'erro quadrático médio alcançou o valor mínimo'
        return condition

    if alpha < p.get_alpha_min():
        condition['result'] = 2
        condition['message'] = 'alpha mínimo alcançado'
        return condition

    if epoch_current == p.get_epochs_max():
        condition['result'] = 3
        condition['message'] = 'número máximo de épocas alcançado'
        return condition

    if (error_list[-1] > error_list[-2] > error_list[-3] > error_list[-4]) and \
     (epoch_current > p.get_epochs_min()):
        condition['result'] = 4
        condition['message'] = 'crescimento sucessivo da taxa de erro quadrática média por 4 épocas'
        return condition

    return condition


def print_title_epoch(epoch, fold_num, message, part_2, descriptor, print_epoch=True):
    """Imprime cabeçalho contendo informações relevantes à rodada"""
    for i in range(100):
        print('-', end='')
    else:
        print()
    if print_epoch:
        print(str.center('FOLD {} - EPOCH {} - {} - {} DELIVERY'.format(fold_num,
            str(epoch).zfill(4), message.upper(), 'SECOND' if part_2 else 'FIRST'), 100))
    else:
        print(str.center('{} - {} DELIVERY'.format(message.upper(),
         'SECOND' if part_2 else 'FIRST'), 100))

    print(str.center('DESCRIPTOR: ' + descriptor, 100))

    message_2 = 'LETTERS: '

    for c in get_classes_letters_list(part_2):
        message_2 += c + ' '

    print(str.center(message_2, 100))

    for i in range(100):
        print('-', end='')
    else:
        print('\n')


def get_letter_from_num(letter_num, part_2):
    """Retorna a letra a partir do número"""
    letters = {}

    for i, letter in (enumerate(get_classes_dict(part_2).values())):
            letters[i] = letter

    return letters[letter_num]


def get_resulting_letter(layer_2, part_2):
    """A partir dos resultados da ultima camada do perceptron,
    retorna a letra obtida no teste"""
    greatest_output = 0
    greatest_row = 0

    for row in range(0, layer_2.shape[0]):
        if layer_2[row][0] > greatest_output:
            greatest_output = layer_2[row][0]
            greatest_row = row

    if greatest_output > 0.1:
        return get_letter_from_num(greatest_row, part_2)
    else:
        return None


def plot_graph(fold_num, errors_test_list, errors_training_list, descriptor, start_algorithm,
l1_neurons, part_2):
    """Função para criação do gráfico de erros"""
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(errors_test_list)
    plt.plot(errors_training_list)
    plt.ylabel('Erros')
    plt.xlabel('Épocas')

    file_command = 'output/error_graph-{desc}-N{hn:03}-P{part}-F{fold}-{datetime}.jpg'.format(
        fold=fold_num, datetime=start_algorithm.strftime('%Y-%m-%d-%H-%M-%S.%f'),
        desc=descriptor, part=2 if part_2 else 1, hn=l1_neurons)

    plt.savefig(file_command)
    plt.close()
