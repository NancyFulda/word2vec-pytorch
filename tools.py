import numpy as np

def load_embeddings(filename="out.vec"):

    f = open(filename, 'r')
    lines = f.readlines()
    embeddings_dict = {}
    for l in lines[1:]:
        tokens = l.split()
        embeddings_dict[tokens[0]] = np.array([float(token) for token in tokens[1:]])
    return embeddings_dict


