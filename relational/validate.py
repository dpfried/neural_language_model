from ntn import NeuralTensorNetwork, TensorLayer
from wordnet_rels import Relationships
import gzip, cPickle
import sys
import numpy as np

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('relationships_path')
    args = parser.parse_args()

    print 'loading model from %s' % args.model_path
    with gzip.open(args.model_path) as f:
        model = cPickle.load(f)

    print 'loading relationships from %s' % args.relationships_path
    with gzip.open(args.relationships_path) as f:
        relationships = cPickle.load(f)

    num_training = int(relationships.N * 0.9)
    training = relationships.data[:num_training]
    testing = relationships.data[num_training:]

    num_testing = testing.shape[0]

    def scored_candidates(a, rel):
        return sorted([(model.test(a, b, rel), b) for b in xrange(len(relationships.synsets))], reverse=True)

    print 'testing'
    try:
        indices = []
        scores = []
        max_scores = []
        for index, (a, true_b, rel) in enumerate(testing):
            sys.stdout.write('\rtesting %d / %d\t %0.2f%%' % (index, num_testing, float(index) / num_testing))
            sys.stdout.flush()
            scored_bs = scored_candidates(a, rel)
            index = [b for (score, b) in scored_bs].index(true_b)
            indices.append(index)
            scores.append(scored_bs[index][0])
            max_scores.append(scored_bs[0][0])

        print '%s out of %s in top 100' % (sum(np.array(indices) < 100), len(indices))
    finally:
        import IPython
        IPython.embed()
