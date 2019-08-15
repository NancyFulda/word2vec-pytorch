import tools

filename='/mnt/pccfs/not_backed_up/nancy/vectors/vectors_window_5/out_3_cleaned.vec.IN'
#filename = '/mnt/pccfs/not_backed_up/nancy/docker/word2vec-pytorch/bse_vectors_window_5/out_iterations_3_cleaned.vec.IN'

print('\n\n', filename)
embeddings_dict = tools.load_embeddings(filename)
tokens = list(embeddings_dict.keys())
vectors = [embeddings_dict[key] for key in tokens]

print('\nnearest')
print(tools.closest_words(embeddings_dict['hot'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['happy'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['beautiful'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['ugly'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['man'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['queen'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['king'], tokens, vectors, n=15))

print('\ntraditional analogies')
print(tools.closest_words(embeddings_dict['king'] - embeddings_dict['man'] + embeddings_dict['woman'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['spain'] - embeddings_dict['madrid'] + embeddings_dict['paris'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['programmer'] - embeddings_dict['man'] + embeddings_dict['woman'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['programmer'] - embeddings_dict['woman'] + embeddings_dict['man'], tokens, vectors, n=15))

print('\nantonym analogies')
print(tools.closest_words(embeddings_dict['hot'] - embeddings_dict['cold'] + embeddings_dict['wet'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['cold'] - embeddings_dict['hot'] + embeddings_dict['wet'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['beautiful'] - embeddings_dict['woman'] + embeddings_dict['man'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['sun'] - embeddings_dict['fire'] + embeddings_dict['moon'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['prince'] - embeddings_dict['youth'] + embeddings_dict['age'], tokens, vectors, n=15))

print('\nother - heat')
print(tools.closest_words(embeddings_dict['ice'] + embeddings_dict['heat'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['water'] - embeddings_dict['heat'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['house'] + embeddings_dict['fire'], tokens, vectors, n=15))

print('\nother - age')
print(tools.closest_words(embeddings_dict['prince'] + embeddings_dict['years'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['father'] + embeddings_dict['old'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['father'] + embeddings_dict['age'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['father'] - embeddings_dict['age'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['father'] + embeddings_dict['youth'], tokens, vectors, n=15))
