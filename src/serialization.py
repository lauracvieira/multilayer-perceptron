#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import functions as f
import imagelib as imagelib
import parameters as p
import _pickle as pickle
import sys

def run():
    # formato do dicionário:
    # { image_name : {  "HOG" : [], 
    #                   "LBP" : []}
    # }

    if 'testes' in sys.argv and 'treinamento' in sys.argv:
        print("Apenas um diretório de execução é aceito. Escolha 'treinamento' ou 'testes'.")
        exit()

    elif 'testes' in sys.argv:
        dataset_type = 'testes'
    elif 'treinamento' in sys.argv:
        dataset_type = 'treinamento'
    else:
        print("Você deve utilizar 'testes' ou 'treinamento' como parâmetro.")
        exit()   


    hog = p.get_parameters('HOG', dataset_type, 'part2' in sys.argv)
    lbp = p.get_parameters('LBP', dataset_type, 'part2' in sys.argv)

    images = {}

    start = datetime.now()
    print("Start time: {}\n".format(start.strftime("%d/%m/%Y %H:%M:%S")))

    try:
        values = {}

        values['HOG'] = imagelib.getHog('./data/img_test.png', hog['descriptor_param_1'],
            hog['descriptor_param_2'], hog['descriptor_param_3'])

        values['LBP'] = imagelib.getLBP('./data/img_test.png', lbp['descriptor_param_1'],
            lbp['descriptor_param_2'])
        
        images['img_test.png'] = values
    except Exception:
        raise
    else:
        print('{}. Teste image img_test.png described by HOG and LBP.'.format(str(0).zfill(5)))

    dataset = f.get_dataset_list(f.get_classes_list(hog.get('workpath')), hog.get('workpath'))
    count = 1

    for line in dataset:
        for name in line:
            if not images.get(name):
                values = {}

                try:
                    values['HOG'] = imagelib.getHog(hog['workpath'] + name, hog['descriptor_param_1'], 
                        hog['descriptor_param_2'], hog['descriptor_param_3'])

                    values['LBP'] = imagelib.getLBP(lbp['workpath'] + name, lbp['descriptor_param_1'], 
                        lbp['descriptor_param_2'])

                    images[name] = values   
                except Exception:
                    raise
                else:
                    print('{}. Image {} described by HOG and LBP.'.format(str(count).zfill(5),
                     name))

                    count += 1

    try:
        with open('./data/{}{}.p'.format(dataset_type, 1 if not hog.get('part_2') else 2), 'wb') as file:
            pickle.dump(images, file)
    except OSError:
        raise OSError
    else:
        end = datetime.now()
        print('\nTotal of images described: {} + test image.\n'.format(str(count - 1).zfill(5)))
        print('File serialized in {}\n'.format(file.name))
        print("Start time: {}\n".format(start.strftime("%d/%m/%Y %H:%M:%S")))
        print("End time: {}\n".format(end.strftime("%d/%m/%Y %H:%M:%S")))
        print('Time running: {}\n'.format(end - start))

if __name__ == "__main__":
    run()       