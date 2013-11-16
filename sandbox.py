from query import query
import gzip, cPickle
import fix_imports
from relational.translation import *
from relational.translational_admm import *
import ngrams

reader = ngrams.NgramReader('/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5', vocab_size=50000)

with gzip.open('/cl/work/dfried/models/factorial/only_syntactic/model-2000.pkl.gz') as f:
    syntactic_model = cPickle.load(f)

with gzip.open('/cl/work/dfried/models/relational/translational_words/model-2000.pkl.gz') as f:
    translational_model = cPickle.load(f)

translational_model.vocabulary = reader.word_array
translational_model._make_lookups()

# nlm_model.vocabulary = reader.word_array
with gzip.open('/cl/work/dfried/models/relational/translational_admm_0.01/model-2000.pkl.gz') as f:
    admm_model = cPickle.load(f)

admm_model.vocabulary = reader.word_array
# admm_model._make_lookups()
