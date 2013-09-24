from model import NLM
import gzip, cPickle
from scipy.spatial.distance import cdist
from nltk.corpus import wordnet as wn
import numpy as np

def most_similar_embeddings(embedding_matrix, index, metric='cosine', top_n=10, **kwargs):
    embedding = embedding_matrix[index]
    C = cdist(embedding[np.newaxis,:], embedding_matrix, metric, **kwargs)
    sims = C[0]
    return [(i, sims[i]) for i in np.argsort(sims)[:top_n]]

def most_similar_synsets(model, synset, metric='cosine', top_n=10, **kwargs):
    index = model.synset_to_symbol[synset]
    mse = most_similar_embeddings(model.synset_embedding_layer.embedding, index, metric=metric, top_n=top_n, **kwargs)
    return [(model.symbol_to_synset[i], sim) for (i, sim) in mse]

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('model_name')
#     args = parser.parse_args()

#     with gzip.open(args.model_name, 'rb') as f:
#         model = cPickle.load(f)
