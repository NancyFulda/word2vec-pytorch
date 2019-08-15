import SIF_embedding
import os
import numpy as np


#for root in ['vectors_window_3','vectors_window_5']:
for root in ['vectors_window_5']:
    for _, _, files in os.walk(root):

        print('processing files in directory ' + root + '/')

        files = ['out_iterations_5_cleaned.vec']
        for filename in files:
            print(filename)
            in_lines = open(root + '/' + filename + '.IN.SIF', 'r').readlines()
            out_lines = open(root + '/' + filename + '.OUT.SIF', 'r').readlines()

            e_in = {}
            e_out = {}

            in_tokens = []
            out_tokens = []

            in_vectors = []
            out_vectors = []

            for l in in_lines[1:]:
                entries = l.split()
                in_tokens.append(entries[0])
                in_vectors.append(np.array([float(entry) for entry in entries[1:]]))
            
            for l in out_lines[1:]:
                entries = l.split()
                out_tokens.append(entries[0])
                out_vectors.append(np.array([float(entry) for entry in entries[1:]]))
            in_vectors = np.array(in_vectors)
            out_vectors = np.array(out_vectors)
            new_vectors = np.multiply(in_vectors/np.mean(in_vectors), out_vectors/np.mean(out_vectors))

            outfile = open(root + '/' + filename+'.SIFMUL2', 'w')
            outfile.write(str(len(new_vectors)) + ' ' + str(len(new_vectors[0])) + '\n')
            for t, v in zip(in_tokens, new_vectors):
                outfile.write(t + ' ' + ' '.join([str(x) for x in v]) + '\n')
            outfile.close()
