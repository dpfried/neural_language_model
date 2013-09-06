from query import load_classifier_and_ngrams

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='zipped model file to read embeddings from')
    parser.add_argument('dump_filename', help='filename to dump the embeddings to')
    parser.add_argument('--ngram_file', help='hd5 file containing the vocabulary')
    parser.add_argument('--unnormalized', action='store_true')
    args = parser.parse_args()

    if args.ngram_file:
        classifier, ngram_reader = load_classifier_and_ngrams(args.model, args.ngram_file)
    else:
        classifier, ngram_reader = load_classifier_and_ngrams(args.model)

    print 'normalized:', not args.unnormalized
    classifier.dump_embeddings(args.dump_filename, ngram_reader.word_array, normalize=not args.unnormalized)
