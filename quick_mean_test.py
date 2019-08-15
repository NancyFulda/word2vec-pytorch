import numpy as np
import SIF_embedding

f=open('vectors_window_5/out_iterations_5_cleaned.vecADD')
lines=f.readlines()

embeddings_dict = {}
tokens = []
vectors = []
for l in lines[1:]:
    entries = l.split()
    tokens.append(entries[0])
    vectors.append(np.array([float(entry) for entry in entries[1:]]))

vectors = np.array(vectors)

SIF_pc = SIF_embedding.compute_pc(vectors)
mean = np.mean(vectors, axis=0)
if np.all(SIF_pc == mean):
    print('yup')
else:
    print('nope')
    print(SIF_pc)
    print(mean)
    print(mean/SIF_pc)

raw_input('>')

SIF_vectors = SIF_embedding.remove_pc(vectors)
mean_vectors = vectors - np.mean(vectors, axis=0)

if np.all(SIF_vectors == mean_vectors):
    print("confirmed: SIF_vectors are just the subtraction of the mean.")
else:
    print('experiment failed')
    print(SIF_vectors[0])
    print(mean_vectors[0])
    print(mean_vectors/SIF_vectors)

