import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, exposure
from skimage.feature import hog
from skimage.io import imread


from os import listdir
from os.path import isfile, join

# Listando as imagens
mypath = "data/dataset1/testes/"
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Abrindo uma imagem
teste = imread(mypath + files[0])
image = color.rgb2gray(teste)

# Tentando usar os parametros padrao
fd, hog_image = hog(
	image,
	orientations=8,
	pixels_per_cell=(8, 8),
	cells_per_block=(16, 16),
	visualise=True
)

# Fazendo o plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

# Imagem de exemplo
ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Imagem hog
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02)) # Rescale histogram for better display
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax2.set_adjustable('box-forced')


plt.show()
