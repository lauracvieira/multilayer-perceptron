from datetime import datetime
import os
import utils as u

executions = [
    ('HOG', '32', ''),  # necessário para comparação com  a primeira fase
    # ('LBP', '160', ''),  # necessário para comparação com primeira fase
    # ('HOG', '32', 'part2'),  # necessário para comparação com  a primeira fase
    # ('LBP', '160', 'part2'),  # necessário para comparação com primeira fase
    # ('HOG', '40', 'part2'),
    # ('LBP', '200', 'part2'),
    # ('HOG', '50', 'part2'),
    # ('LBP', '240', 'part2'),
    # ('HOG', '60', 'part2'),
    # ('LBP', '280', 'part2'),
    # ('HOG', '70', 'part2'),
    # ('LBP', '320', 'part2'),
    # ('HOG', '80', 'part2'),
    # ('LBP', '360', 'part2'),
    # ('HOG', '100', 'part2'),
    # ('LBP', '400', 'part2')
]

if __name__ == '__main__':
    start = datetime.now()
    hog_count = 1
    lbp_count = 1

    for i, e in enumerate(executions):
        command = 'python3 src/cross-validation.py '
        command += '{desc} {neurons:3} {p_arg:5} > ./output/{desc}-N{hn:03}-P{part}-log-{datetime}_{count:02}.txt &'.format(
            desc=e[0], neurons=e[1], p_arg=e[2], part=2 if e[2] == 'part2' else 1,
            datetime=start.strftime('%Y-%m-%d-%H-%M'), count=hog_count if e[0] == 'HOG' else lbp_count,
            hn=int(e[1]))

        if e[0] == 'HOG':
            hog_count += 1
        else:
            lbp_count += 1

        os.system(command)
        print('{}. Running: {}'.format(str(i + 1).zfill(2), command))
