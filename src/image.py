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

def getImages(path):
	files = []

   	for f in listdir(path):
   		file = join(path, f)
   		if isfile(file):
   			files.append(getImage(file))
	return files;

def calculateHog(image):
	# http://scikit-image.org/docs/dev/api/skimage.feature.html#hog
	return hog( 
		image,
		orientations=9,
		pixels_per_cell=(8, 8),
	    cells_per_block=(1, 1),
	    block_norm='L2-Hys'
	)

def calculateLBP(image):
	# http://scikit-image.org/docs/dev/api/skimage.feature.html#local-binary-pattern
	return local_binary_pattern(
		image,
		P = 8,
		R = 2,
		method='default'
	)