import SIF_embedding
import os
import numpy as np


#for root in ['vectors_window_3','vectors_window_5']:
for root in ['vectors_window_5']:
    for _, _, files in os.walk(root):

        print('processing files in directory ' + root + '/')
        for filename in files:

            if filename[-5:] != '.MUL2':
                continue
    
            print(filename)
            lines = open(root + '/' + filename, 'r').readlines()
            embeddings_dict = {}
            tokens = []
            vectors = []
            for l in lines[1:]:
                entries = l.split()
                tokens.append(entries[0])
                vectors.append(np.array([float(entry) for entry in entries[1:]]))
            
            new_vectors = SIF_embedding.remove_pc(np.array(vectors))

            outfile = open(root + '/' + filename+'.SIF', 'w')
            outfile.write(str(len(new_vectors)) + ' ' + str(len(new_vectors[0])) + '\n')
            for t, v in zip(tokens, new_vectors):
                outfile.write(t + ' ' + ' '.join([str(x) for x in v]) + '\n')
            outfile.close()
