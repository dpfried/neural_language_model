import os
import sys
from utils import models_in_folder

def models_to_delete(directory, retain_every=10):
    models = models_in_folder(directory)

    if not models:
        print 'no models found'
        return {}, set()

    to_keep = set([n for n in models
                   if n % retain_every == 0])
    # don't delete the first one, so we can use it as a reference point
    to_keep.add(min(models))

    # the last one may be currently being written
    to_keep.add(max(models))

    return dict((model, path)
            for model, path in models.items()
            if model not in to_keep), to_keep

def delete_models(models):
    for model_num, path in models.items():
        sys.stdout.write('\rremoving model %s' % path)
        sys.stdout.flush()
        os.remove(path)

# one off script used to clean out any models in the passed folders that aren't
# multiples of 10
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_directories', nargs='+')
    parser.add_argument('--retain_every', type=int, default=10)
    args = parser.parse_args()

    for directory in args.model_directories:
        to_delete, to_keep_nums = models_to_delete(directory, retain_every=args.retain_every)
        print directory
        if not to_delete:
            print 'no models to delete (%i present)' % len(to_keep_nums)
            continue
        print 'would delete %i models, leaving %i' % (len(to_delete), len(to_keep_nums))
        print 'do it? (yes to continue)',
        if raw_input() == 'yes':
            delete_models(to_delete)
        print
