#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# arquivo de parâmetros

# mlp
__EPOCHS = 50
__ALPHA = 0.5

# hog: parâmetros
__HIDDEN_NEURONS_LBP = 32
__HOG_ORIENTATIONS = 9
__HOG_PIXELS_PER_CELL = 8
__HOG_CELLS_PER_BLOCK = 1

# lbp: parâmetros
__HIDDEN_NEURONS_HOG = 160
__LBP_POINTS = 24
__LBP_RADIUS = 8

def get_parameters(descriptor, dataset_type, part_2):
    """Retorna um dicionário contendo os parâmetros de configuração do algoritmo e dos descritores""" 
    if descriptor not in ['HOG', 'LBP']:
        return None

    p = {}

    p['descriptor'] = descriptor
    p['epochs'] = __EPOCHS 
    p['alpha'] = __ALPHA 
    p['part_2'] = part_2
    p['dataset_type'] = dataset_type
    p['hidden_neurons'] = __HIDDEN_NEURONS_HOG if descriptor == 'HOG' else __HIDDEN_NEURONS_LBP
    p['descriptor_param_1'] = __HOG_ORIENTATIONS if descriptor == 'HOG' else __LBP_POINTS
    p['descriptor_param_2'] = __HOG_PIXELS_PER_CELL if descriptor == 'HOG' else __LBP_RADIUS
    p['descriptor_param_3'] = __HOG_CELLS_PER_BLOCK if descriptor == 'HOG' else 0

    if part_2:
        p['workpath'] = './data/dataset2/' + dataset_type + '/'
    else:
        p['workpath'] = './data/dataset1/' + dataset_type + '/'

    return p


    
