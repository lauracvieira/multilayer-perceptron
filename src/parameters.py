#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# arquivo de parâmetros

# mlp
EPOCHS = 5000
HIDDEN_NEURONS = 32
ALPHA = 0.5

# hog: parâmetros
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = 8
HOG_CELLS_PER_BLOCK = 1

# lbp: parâmetros
LBP_POINTS = 24
LBP_RADIUS = 8

# entrega 1 = False, entrega 2 = True
PART_2 = not True

# diretório dos datasets
if PART_2:
	WORKPATH = './data/dataset2/treinamento/'
else:
	WORKPATH = './data/dataset1/treinamento/'

def get_parameters(descriptor):
    """Retorna um dicionário contendo os parâmetros de configuração do algoritmo e dos descritores""" 
    if descriptor not in ['HOG', 'LBP']:
        return None

    p = {}

    p['descriptor'] = descriptor
    p['epochs'] = EPOCHS 
    p['hidden_neurons'] = HIDDEN_NEURONS
    p['alpha'] = ALPHA 
    p['workpath'] = WORKPATH
    p['part_2'] = PART_2

    if descriptor == 'HOG':
        p['descriptor_param_1'] = HOG_ORIENTATIONS
        p['descriptor_param_2'] = HOG_PIXELS_PER_CELL
        p['descriptor_param_3'] = HOG_CELLS_PER_BLOCK
    else:
        p['descriptor_param_1'] = LBP_POINTS
        p['descriptor_param_2'] = LBP_RADIUS
        p['descriptor_param_3'] = 0

    return p


    
