#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import functions as f
import imagelib as i
import parameters as p
from datetime import datetime

def run():
	# formato do dicionário:
	# { image_name : {	"hog" : [], 
	# 					"lbp" : []}
	# }

	dataset = f.get_dataset(f.get_classes())
	count = 1
	images = {}
	start = datetime.now()

	print("Start time: {}\n".format(start.strftime("%d/%m/%Y %H:%M:%S")))

	for line in dataset:
		for name in line:
			if not images.get(name):
				values = {}

				try:
					values['hog'] = i.getHog(p.WORKPATH + name, p.HOG_ORIENTATIONS, 
						p.HOG_PIXELS_PER_CELL, p.HOG_CELLS_PER_BLOCK)

					values['lbp'] = i.getLBP(p.WORKPATH + name, p.LBP_POINTS, 
						p.LBP_RADIUS)

					images[name] = values	
				except Exception:
					raise
				else:
					print('{}. Image {} described by HOG and LBP.'.format(str(count).zfill(5),
					 name))

					count += 1

	try:
		with open('./data/dataset.p', 'wb') as file:
			pickle.dump(images, file)
	except OSError:
		raise OSError
	else:
		print('\nFile serialized in ./data/{}\n'.format(file.name))
		end = datetime.now()
		print("Start time: {}\n".format(start.strftime("%d/%m/%Y %H:%M:%S")))
		print("End time: {}\n".format(end.strftime("%d/%m/%Y %H:%M:%S")))
		print('Time running: {}\n'.format(end - start))

if __name__ == "__main__":
	run()		