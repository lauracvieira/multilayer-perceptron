#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# arquivo de parâmetros

# mlp
EPOCHS = 5000
HIDDEN_NEURONS = 160
ALPHA = 0.5

# descritor utilizado
DESCRIPTOR = 'LBP'

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