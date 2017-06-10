#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import utils as u
import imagelib as imagelib
import parameters as p
import sqlite3
import progressbar
import sys
import time

def run():
    if sys.platform == 'win32':
        barvalue = '#'
    else:
        bar_value  = '█'


    start = datetime.now()
    path = './data/database.db'
    conn = sqlite3.connect(path)
    
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS treinamento (
                                                                                                            image_name TEXT PRIMARY KEY, 
                                                                                                            HOG BLOB , 
                                                                                                            LBP BLOB )''')

    c.execute('''CREATE TABLE IF NOT EXISTS testes (       
                                                                                                    image_name TEXT PRIMARY KEY, 
                                                                                                    HOG BLOB , 
                                                                                                    LBP BLOB )''')

    hog = p.get_parameters('HOG', True)
    lbp = p.get_parameters('LBP', True)

    try:
        values = {}

        values['HOG'] = imagelib.getHog('./data/img_test.png', hog['descriptor_param_1'],
            hog['descriptor_param_2'], hog['descriptor_param_3'])

        values['LBP'] = imagelib.getLBP('./data/img_test.png', lbp['descriptor_param_1'],
            lbp['descriptor_param_2'])

        c.execute('INSERT OR IGNORE INTO treinamento VALUES (?, ?, ?)', ['img_test.png', values['HOG'], values['LBP']])
        
    except Exception:
        raise

    dataset = u.get_dataset_list(u.get_classes_list('./data/dataset2/treinamento'), './data/dataset2/treinamento')
    count = 1

    for line in dataset:
        widgets = ['Training Letter: {} | '.format(u.get_letter(line[0], True)), progressbar.Percentage(), 
        ' (', progressbar.Counter(), ' of ',  str(len(line)), ') ', progressbar.Bar(bar_value), '  ',progressbar.ETA()]
        with progressbar.ProgressBar(widgets=widgets, max_value=len(line)) as bar:
            for i, name in enumerate(line):
                values = {}

                try:
                    values['HOG'] = imagelib.getHog(hog['workpath'] + name, hog['descriptor_param_1'], 
                        hog['descriptor_param_2'], hog['descriptor_param_3'])

                    values['LBP'] = imagelib.getLBP(lbp['workpath'] + name, lbp['descriptor_param_1'], 
                        lbp['descriptor_param_2'])

                    c.execute('INSERT OR IGNORE INTO treinamento VALUES (?, ?, ?)', [name, values['HOG'], values['LBP']])
                except Exception:
                    raise

                count += 1
                bar.update(i)

        conn.commit()

    dataset = u.get_dataset_list(u.get_classes_list('./data/dataset2/testes'), './data/dataset2/testes')

    for line in dataset:
        widgets = ['Testing Letter: {} | '.format(u.get_letter(line[0], True)), progressbar.Percentage(), 
        ' (', progressbar.Counter(), ' of ',  str(len(line)), ') ', progressbar.Bar(bar_value), '  ',progressbar.ETA()]
        with progressbar.ProgressBar(widgets=widgets, max_value=len(line)) as bar:
            for i, name in enumerate(line):
                values = {}

                try:
                    values['HOG'] = imagelib.getHog(hog['testpath'] + name, hog['descriptor_param_1'], 
                        hog['descriptor_param_2'], hog['descriptor_param_3'])

                    values['LBP'] = imagelib.getLBP(lbp['testpath'] + name, lbp['descriptor_param_1'], 
                        lbp['descriptor_param_2'])

                    c.execute('INSERT OR IGNORE INTO testes VALUES (?, ?, ?)', [name, values['HOG'], values['LBP']])
                except Exception:
                    raise

                count += 1
                bar.update(i)         
        conn.commit()
    c.close()
    end = datetime.now()
    print('\nTotal of images described: {} + test image.'.format(str(count - 1).zfill(5)))
    print('Database in {}\n'.format(path))
    print("Start Time: {}".format(start.strftime("%d/%m/%Y %H:%M:%S")))
    print("End Time: {}".format(end.strftime("%d/%m/%Y %H:%M:%S")))
    print('Total Elapsed Time: {}\n'.format(end - start))

if __name__ == "__main__":
    run()       