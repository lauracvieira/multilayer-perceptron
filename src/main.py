from image import getImages, calculateHog, calculateLBP

images = getImages("data/dataset1/treinamento/")

for image in images:
	hogResult = calculateHog(image)
	lbpResult = calculateLBP(image)
