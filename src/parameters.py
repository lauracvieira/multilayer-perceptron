#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# arquivo de parâmetros

# mlp
__EPOCHS_MAX = 500
__EPOCHS_MIN = 10
__ALPHA = 0.5
__BIAS = 1

# stop condition
__ERROR_MIN = 0.015
__ALPHA_MIN = 0.001

# hog: parâmetros
__HIDDEN_NEURONS_HOG = 32
__HOG_ORIENTATIONS = 9
__HOG_PIXELS_PER_CELL = 8
__HOG_CELLS_PER_BLOCK = 1

# lbp: parâmetros
__HIDDEN_NEURONS_LBP = 160
__LBP_POINTS = 24
__LBP_RADIUS = 8


def get_parameters(descriptor, part_2, hidden_neurons=None, output_directory=None):
    """Retorna um dicionário contendo os parâmetros de
    configuração do algoritmo e dos descritores"""
    if descriptor not in ['HOG', 'LBP']:
        return None

    p = {}

    p['descriptor'] = descriptor
    p['epochs'] = __EPOCHS_MAX
    p['alpha'] = __ALPHA
    p['bias'] = __BIAS
    p['part_2'] = part_2

    if hidden_neurons is not None:
            p['hidden_neurons'] = int(hidden_neurons)
    else:
        p['hidden_neurons'] = __HIDDEN_NEURONS_HOG if descriptor == 'HOG' else __HIDDEN_NEURONS_LBP

    p['descriptor_param_1'] = __HOG_ORIENTATIONS if descriptor == 'HOG' else __LBP_POINTS
    p['descriptor_param_2'] = __HOG_PIXELS_PER_CELL if descriptor == 'HOG' else __LBP_RADIUS
    p['descriptor_param_3'] = __HOG_CELLS_PER_BLOCK if descriptor == 'HOG' else 0
    p['workpath'] = './data/dataset{}/treinamento/'.format(2 if part_2 else 1)
    p['testpath'] = './data/dataset{}/testes/'.format(2 if part_2 else 1)
    p['output'] = output_directory

    return p


def get_error_min():
    return __ERROR_MIN


def get_epochs_min():
    return __EPOCHS_MIN


def get_epochs_max():
    return __EPOCHS_MAX


def get_alpha_min():
    return __ALPHA_MIN

    