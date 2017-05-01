import numpy as np

def initnw(inputs,hidden):
    """
    Nguyen-Widrow initialization function
    :Parameters:
        layer: core.Layer object
            Initialization layer
    """
    ci = inputs
    cn = hidden
    w_fix = 0.7 * cn ** (1. / ci)
    w_rand = np.random.rand(cn, ci) * 2 - 1
    # Normalize
    if ci == 1:
        w_rand /= np.abs(w_rand)
    else:
        w_rand *= np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(cn, 1))

    w = w_fix * w_rand
    b = np.array([0]) if cn == 1 else w_fix * \
        np.linspace(-1, 1, cn) * np.sign(w[:, 0])

    return w

def nguyen():
    neww= np.array([initnw(3,1),initnw(3,1),initnw(3,1),initnw(3,1)])
    print("Valores Nguyen: \n\n{0}".format(neww))
    return neww;

if __name__ == "__main__":
    nguyen()
