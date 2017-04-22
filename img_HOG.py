import numpy as np #array manipulation
import matplotlib
import cv2

winSize = (128,128) #image size
blockSize = (16,16)
blockStride = (8,8) #normally 50% of the blocksize, multiple of cell size
cellSize = (8,8) 
nbins = 9 #default
derivAperture = 1 #default
winSigma = -1 #default
histogramNormType = 0 #default
L2HysThreshold = 0.2 #default
gammaCorrection = True #default
nlevels = 64 #default
signedGradients = True

def main():
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)
	im = cv2.imread('img_test.png')
	descriptor = hog.compute(im)


if __name__ == "__main__":
	main()