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
    #this doesn't work because the vectors aren't normalized
    #distances = 1 - np.dot(vectors,vec.T)/(np.linalg.norm(vectors) * np.linalg.norm(vec))

    #but this does
    distances = spatial.distance.cdist(vectors, [vec], metric='cosine')

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

    #this doesn't work because vectors are not normalized
    #distances = 1 - np.dot(vectors,vec.T)/np.linalg.norm(vec)

    #but this does
    distances = spatial.distance.cdist(vectors, [vec], metric='cosine')

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
        #whr = np.where(filtered_vectors == v[i])
        #print(whr)
        #print(filtered_vectors[0].shape)
        #print(type(filtered_vectors[0]))
        #input('>')
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
        print('num correct %d/%d, %f' % (num_correct, count, float(num_correct)/count))

        results_dict[data[0]] = str(num_correct) + ' correct, ' + str(count) + ' total, ' + str(100 * float(num_correct)/count) + '%, omitted ' + str(num_omitted)

    print('\n\nGOOGLE ANALOGY TEST SET')
    pp = pprint.PrettyPrinter()
    pp.pprint(results_dict)
    print('Composite %f%s' % (100 * float(total_correct)/total_count, '%'))
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
        tokens = embeddings_dict.keys()
        vectors = [embeddings_dict[key] for key in embeddings_dict.keys()]
        run_google_evaluation(embeddings, tokens, vectors)
        run_BATS_evaluation(embeddings, tokens, vectors)
        run_SAT_evaluation(embeddings, tokens, vectors)

if __name__=='__main__':
    #embeddings_dict = load_embeddings('bse_vectors_DIM_50_window_5_small_corpus/out_iterations_5.vec.ADD.SIF')
    embeddings_dict = load_embeddings('bse_constraint_vectors_window_5/out_iterations_2_cleaned.vec.OUT.SIF')
    tokens = list(embeddings_dict.keys())
    vectors = [embeddings_dict[key] for key in tokens]
    #tokens = tokens[:10000]
    #vectors = vectors[:10000]
    run_google_evaluation(embeddings_dict, tokens, vectors)
