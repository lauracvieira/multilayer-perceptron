import numpy as np

from skimage import color
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.io import imread

from os import listdir
from os.path import isfile, join

def crop(img, new_height, new_width):
   width =  np.size(img, 1)
   height =  np.size(img, 0)

   left = int(np.ceil((width - new_width)/2))
   top = int(np.ceil((height - new_height)/2))
   right = int(np.floor((width + new_width)/2))
   bottom = int(np.floor((height + new_height)/2))

   cImg = img[top:bottom, left:right]
   return cImg

def getImage(file):
	image = imread(file)
	image = crop(image, 64, 64)
	image = color.rgb2gray(image)
	return image

def getImages(path):
	files = []

	for f in listdir(path):
		file = join(path, f)
		if isfile(file):
			files.append(getImage(file))

	return files

def calculateHog(image, hog_orientations, hog_pixels_per_cell, hog_cell_per_block):
	# http://scikit-image.org/docs/dev/api/skimage.feature.html#hog
	return hog( 
		image,
		orientations= hog_orientations,
		pixels_per_cell=(hog_pixels_per_cell, hog_pixels_per_cell),
	    cells_per_block=(hog_cell_per_block, hog_cell_per_block),
	    block_norm='L2-Hys'
	)

def calculateLBP(image, lbp_points, lbp_radius):
	# http://scikit-image.org/docs/dev/api/skimage.feature.html#local-binary-pattern
	return local_binary_pattern(
		image,
		P = lbp_points,
		R = lbp_radius,
		method='default'
	)

def getHog(image_name, hog_orientations, hog_pixels_per_cell, hog_cell_per_block):
	image = getImage(image_name)
	return calculateHog(image, hog_orientations, hog_pixels_per_cell, hog_cell_per_block)

def getLBP(image_name, lbp_points, lbp_radius):
	image = getImage(image_name)
	return calculateLBP(image, lbp_points, lbp_radius)

