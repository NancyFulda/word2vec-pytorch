import numpy as np
import pprint
from scipy import spatial
from tqdm import tqdm

def load_embeddings(filename="out.vec"):

    f = open(filename, 'r')
    lines = f.readlines()
    embeddings_dict = {}
    for l in lines[1:]:
        tokens = l.split()
        embeddings_dict[tokens[0]] = np.array([float(token) for token in tokens[1:]])
    return embeddings_dict

def closest_words(vec, tokens, vectors, n=5):
    distances = 1 - np.dot(vectors,vec.T)/np.linalg.norm(vec)

    closest_words = []
    for i in range(n):
        min_dist = np.nanmin(distances)
        index = np.where(distances == min_dist)[0][0]
        closest_word = tokens[index]
        closest_vector = vectors[index]

        distances[index] = 1000
        closest_words.append(closest_word)

    return closest_words

def test_analogy(words, embeddings_dict, n=5):
    tokens = list(embeddings_dict.keys())
    vectors = [embeddings_dict[key] for key in tokens]
    
    vecs = [embeddings_dict[w.lower()] for w in words]
    analogy = vecs[1] - vecs[0] + vecs[2]

    return closest_words(analogy, tokens, vectors, n)

def find_closest_point(vec, tokens, vectors):

    distances = 1 - np.dot(vectors,vec.T)/np.linalg.norm(vec)
    min_dist = np.nanmin(distances)
    index = np.where(distances == min_dist)[0][0]
    closest_word = tokens[index]
    closest_vector = vectors[index]

    return closest_vector

def compute_analogy(v, tokens, vectors):
    vec = v[1] - v[0] + v[2]

    filtered_tokens = tokens[:]
    filtered_vectors = vectors[:]
    for i in range(3):
        index = np.where(filtered_vectors == v[i])[0][0]
        del filtered_tokens[index]
        del filtered_vectors[index]

    return all(find_closest_point(vec, filtered_tokens, filtered_vectors) == v[3])

def run_google_evaluation(embeddings_dict, tokens, vectors):

    f = open('data/google_analogy_test_set.txt','r')
    categories = f.read().split(':')[1:]
    results_dict = {}
    print('%d categories' % (len(categories)))

    total_correct = 0
    total_count = 0
    total_omitted = 0
    for cat in categories:
        data = cat.split('\n')

        category_name = data[0]
        count = len(data[1:])
        num_correct = 0
        num_omitted = 0
    
        print(category_name)
        #for analogy in tqdm(data[1:]):
        for analogy in data[1:]:
            words = analogy.split()
            if len(words) == 0:
                continue

            #try:
            analogy_vectors = [embeddings_dict[w.lower()] for w in words]
    
            num_correct += compute_analogy(analogy_vectors, tokens, vectors)        
            #except Exception as e:
            #    print(e)
            #    num_omitted += 1
        total_correct += num_correct
        total_count += count
        total_omitted += num_omitted
        print('num correct %d/%d' % (num_correct, count))

        results_dict[data[0]] = str(num_correct) + ' correct, ' + str(count) + ' total, ' + str(100 * num_correct/count) + '%, omitted ' + str(num_omitted)

    print('\n\nGOOGLE ANALOGY TEST SET')
    pp = pprint.PrettyPrinter()
    pp.pprint(results_dict)
    print('Composite %.2f%s' % (100 * total_correct/total_count, '%'))
    print('Omissions %d' % (total_omitted))

def run_BATS_evaluation(embeddings_dict, tokens, vectors):
    pass

def run_SAT_evaluation(embeddings_dict, tokens, vectors):
    pass

def run_evaluations(embeddings_dict, tokens, vectors):

    filenames = ['out.vec.IN']
    
    for f in filenames:
        print('\n loading %s' % (f))
        embeddings = load_embeddings(f)
        tokens = sorted(embeddings_dict.keys())
        vectors = [embeddings_dict[key] for key in sorted(embeddings_dict.keys())]
        run_google_evaluation(embeddings, tokens, vectors)
        run_BATS_evaluation(embeddings, tokens, vectors)
        run_SAT_evaluation(embeddings, tokens, vectors)

if __name__=='__main__':
    embeddings_dict = load_embeddings('out.vec.IN')
    tokens = list(embeddings_dict.keys())
    vectors = [embeddings_dict[key] for key in tokens]
    short_tokens = tokens[:10000]
    short_vectors = vectors[:10000]
    run_google_evaluation(embeddings_dict, short_tokens, short_vectors)
