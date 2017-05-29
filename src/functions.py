#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import parameters as p

def create_directories(directories):
    """Criação dos diretórios"""
    for directory in directories:
        try:
            os.stat('./' + directory)
        except:
            os.mkdir('./' + directory)


def get_classes():
    """Listagem das classes existentes no diretório"""
    classes = list()

    for f in os.listdir(p.WORKPATH):
        if f[:8] not in classes:
            classes.append(f[:8])

    return classes


def get_dataset(classes):
    """Reúne os arquivos de cada classe em uma lista de listas"""
    dataset = list()

    for i in range(len(classes)):
        files = [f for f in os.listdir(p.WORKPATH) if f.startswith(classes[i])]
        dataset.append(files)

    return dataset


def get_classes_dict(part_2):
    """Retorna um dicionário contendo as letras das classes
        correspondentes ao nome físico dos arquivos"""
    if part_2:
        classes = { 'train_41': 'A', 
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
                    'train_5a': 'Z'}
    else:
        classes = { 'train_53': 'S', 
                    'train_58': 'X',
                    'train_5a': 'Z'}
    return classes


def get_classes_list(part_2):
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


def serialize_model(fold_num, weight_0, weight_1):
    """Serialiização dos pesos no arquivo output/model.dat"""
    data = (weight_0, weight_1)
    f = open('./output/model_{0}.dat'.format(fold_num), "wb")
    pickle.dump(data, f)
    f.close()


def de_serialize_model():
    """Resgata os dados serializados no arquivo model.dat"""
    f = open('./output/model.dat', "rb")
    data = pickle.load(f)
    f.close()

    return data


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


def error_list_update(error, error_list):
    """Atualiza a lista de erros de forma circular para que ela contenha os erros das últimas 5 épocas"""
    if len(error_list) == 5:
        # remove o erro mais antigo da lista
        del error_list[0]

    # adiciona o erro da última época na última posição
    error_list.append(error)

    return error_list


def stop_condition(error_list):
    """Informa quando o erro é crescente por 5 épocas consecutivas"""
    if len(error_list) < 5:
        return False

    return error_list[4] > error_list[3] > error_list[2] > error_list[1] > error_list[0]


def print_title_epoch(epoch, message, part_2, descriptor):
    """Imprime cabeçalho contendo informações relevantes à rodada"""
    for i in range(100):
        print('-', end='')
    else:
        print()

    print(str.center('EPOCH {0} - {1} - {2} DELIVERY'.format(str(epoch).zfill(4), 
        message.upper(), 'SECOND' if part_2 else 'FIRST'), 100))

    print(str.center('DESCRIPTOR: ' + descriptor, 100))
    
    message_2 = 'LETTERS: '

    for c in get_classes_list(part_2):
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
    """A partir dos resultados da ultima camada do perceptron, retorna a letra obtida no teste"""
    greatest_output = 0
    greatest_row = 0

    for row in range(0, layer_2.shape[0]):
        if layer_2[row][0] > greatest_output:
            greatest_output = layer_2[row][0]
            greatest_row = row

    if greatest_output > 0.7:
        return get_letter_from_num(greatest_row, part_2)
    else:
        return None