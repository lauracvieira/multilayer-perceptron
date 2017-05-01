import numpy as np

def initnw(inputs,outputs):
    ci = inputs
    cn = outputs
    w_fix = 0.7 * cn ** (1. / ci)
    w_rand = np.random.rand(cn, ci) * 2 - 1
    # Normalize
    if ci == 1:
        w_rand /= np.abs(w_rand)
    else:
        w_rand *= np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(cn, 1))

    w = w_fix * w_rand
    return w

def nguyen(inputs,outputs):
    neww = []
    neww = initnw(inputs,outputs)
    return neww.T;
