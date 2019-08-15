import tools

filename='/mnt/pccfs/not_backed_up/nancy/vectors/vectors_window_5/out_3_cleaned.vec.IN'
#filename = '/mnt/pccfs/not_backed_up/nancy/docker/word2vec-pytorch/bse_vectors_window_5/out_iterations_3_cleaned.vec.IN'

print('\n\n' + filename)
embeddings_dict = tools.load_embeddings(filename)
tokens = list(embeddings_dict.keys())
vectors = [embeddings_dict[key] for key in tokens]

print('\nnearest')
for word in ['hot','happy','beautiful','ugly','man','queen','perspective']:
    print('\n' + word)
    print(tools.closest_words(embeddings_dict[word], tokens, vectors, n=15))

"""print('\ntraditional analogies')
print(tools.closest_words(embeddings_dict['king'] - embeddings_dict['man'] + embeddings_dict['woman'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['spain'] - embeddings_dict['madrid'] + embeddings_dict['paris'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['programmer'] - embeddings_dict['man'] + embeddings_dict['woman'], tokens, vectors, n=15))
print(tools.closest_words(embeddings_dict['programmer'] - embeddings_dict['woman'] + embeddings_dict['man'], tokens, vectors, n=15))"""

print('\nantonym analogies')
print('\nhot-cold+wet')
print(tools.closest_words(embeddings_dict['hot'] - embeddings_dict['cold'] + embeddings_dict['wet'], tokens, vectors, n=15))
print('\ncold-hot+wet')
print(tools.closest_words(embeddings_dict['cold'] - embeddings_dict['hot'] + embeddings_dict['wet'], tokens, vectors, n=15))
print('\nhandsome-man+woman')
print(tools.closest_words(embeddings_dict['handsome'] - embeddings_dict['man'] + embeddings_dict['woman'], tokens, vectors, n=15))
print('\nwar-peace+happy')
print(tools.closest_words(embeddings_dict['war'] - embeddings_dict['peace'] + embeddings_dict['happy'], tokens, vectors, n=15))

print('\nother - heat')
print('\nice+heat')
print(tools.closest_words(embeddings_dict['ice'] + embeddings_dict['heat'], tokens, vectors, n=15))
print('\nsnow+heat')
print(tools.closest_words(embeddings_dict['ice'] + embeddings_dict['heat'], tokens, vectors, n=15))
print('\nwater-heat')
print(tools.closest_words(embeddings_dict['water'] - embeddings_dict['heat'], tokens, vectors, n=15))
print('\nfire+big')
print(tools.closest_words(embeddings_dict['fire'] + embeddings_dict['big'], tokens, vectors, n=15))

print('\nother - age')
print('\nchild+old')
print(tools.closest_words(embeddings_dict['child'] + embeddings_dict['age'], tokens, vectors, n=15))
print('\ngrandfather-old')
print(tools.closest_words(embeddings_dict['grandfather'] - embeddings_dict['old'], tokens, vectors, n=15))
print('\nman+wisdom')
print(tools.closest_words(embeddings_dict['man'] + embeddings_dict['wisdom'], tokens, vectors, n=15))

print('\ninversions')
print('-age')
print(tools.closest_words(-1*embeddings_dict['age'], tokens, vectors, n=15))
print('-rich')
print(tools.closest_words(-1*embeddings_dict['rich'], tokens, vectors, n=15))
print('-happy')
print(tools.closest_words(-1*embeddings_dict['happy'], tokens, vectors, n=15))
