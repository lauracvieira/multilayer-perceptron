from datetime import datetime
import os
import utils as u

# Para cada caso de teste, insira uma tupla no formato (<descritor>, <neurônios>, <parte>), sendo:
# <descritor>:    'HOG' ou 'LBP'
# <neurônios>:    Número de neurônios na camada escondida
# <parte>:            'part2' se for a parte 2, se for a parte, apenas '' (dois apóstrofos)

# Exemplo:
# executions = [
#     # ('HOG', '32', '')               <- HOG, 32 neurônios, parte 1
#     # ('LBP', '160', 'part2')    <- LBP, 160 neurônios, parte 2
# ]
# O número de tuplas de teste é ilimitado, então use de acordo com sua necessidade.
# Ao definir todos os casos, execute no terminal: python3 src/run.py

executions = [
    ('HOG', '32', '') 
    ('LBP', '160', '') 
]

if __name__ == '__main__':
    start = datetime.now()

    for run_num, e in enumerate(executions):
        directory = 'output/{desc}-N{hn:03}-P{part}-{datetime}/'.format(
            desc=e[0], neurons=e[1], part=2 if 'part2' in e else 1,
            datetime=start.strftime('%Y-%m-%d-%H-%M'), hn=int(e[1]))

        u.create_directories(['output', directory])
        command = 'python3.6 src/cross-validation.py '
        command += '{desc} {neurons:3} {part:5} {directory} > {directory}log.txt &'.format(desc=e[0],
            neurons=e[1], part=e[2], directory=directory)

        os.system(command)
        print('{}. Running: {}'.format(str(run_num + 1).zfill(2), command))
