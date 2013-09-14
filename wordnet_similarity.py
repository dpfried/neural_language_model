from nltk.corpus import wordnet as wn
import itertools
import numpy as np
import sys
import math

def pos_depth(synset):
    try:
        if synset.pos not in synset._wordnet_corpus_reader._max_depth:
            synset._wordnet_corpus_reader._compute_max_depth(synset.pos, True)

        return synset._wordnet_corpus_reader._max_depth[synset.pos]
    except:
        print synset

def lch_similarity(wunsch_paths, synset1, synset2):
        """
        Leacock Chodorow Similarity:
        Return a score denoting how similar two word senses are, based on the
        shortest path that connects the senses (as above) and the maximum depth
        of the taxonomy in which the senses occur. The relationship is given as
        -log(p/2d) where p is the shortest path length and d is the taxonomy
        depth.

        :return: A score denoting the similarity of the two ``Synset`` objects,
            normally greater than 0. None is returned if no connecting path
            could be found. If a ``Synset`` is compared with itself, the
            maximum score is returned, which varies depending on the taxonomy
            depth.
        """

        if synset1.pos != synset2.pos:
            return None

        distance = wunsch_paths.synset_path_distance(synset1, synset2)

        depth = pos_depth(synset1)

        if distance is None or distance < 0 or depth == 0:
            return None
        return -math.log((distance + 1) / (2.0 * depth + 2))

def distance_norm(reference_synset):
    depth = pos_depth(reference_synset)
    if depth == 0:
        return None
    return -math.log(1. / (2.0 * depth + 2))

def scaled_lch_similarity(wunsch_paths, synset1, synset2):
    lch_sim = lch_similarity(wunsch_paths, synset1, synset2)
    if lch_sim is None:
        return None
    # since they both have the same POS both should have the same
    # distance norm (if they don't have the same POS lch_similarity
    # is undefined, returns None, and we've already terminated
    dist_norm = distance_norm(synset1)
    if dist_norm is None:
        return None
    return 2 * (lch_sim / dist_norm) - 1

class WunschNode(object):
    INNER, ROOT, TREE, LEAF = range(1, 5)
    def __init__(self, index, synset):
        #  a unique identifier for the node within the entire graph
        self.index = index
        # the L{Synset} this node contains
        self.synset = synset
        # either INNER, ROOT, TREE, or LEAF
        self.class_ = None
        # if this is a TREE or ROOT node, the root node of the tree it is
        # contained in
        self.root_node = None
        # nodes containing the hypernyms of this node's synset
        self.parents = set()
        # nodes containing the synsets of which this node's synset is
        # a hypernym
        self.children = set()
        # if this node is an INNER node, the index of the node
        # in inner subgraph
        self.inner_index = None

        self.tree_depth = None

    def __repr__(self):
        return '<%i: %s>' % (self.index, self.synset)

    def get_single_parent(self):
        """
        assuming that his node has only one parent, return that parent
        """
        for parent in self.parents:
            return parent

    def ancestors(self):
        """
        return a set of the transitive closure of the parent relationship on
        this node (i.e.  the set of all parents, of parents of parents, etc)
        """
        if not self.parents:
            return set()
        else:
            return reduce(lambda acc, update: acc.union(update),
                          (parent.ancestors() for parent in self.parents),
                          self.parents)

    def propagate_root(self, root_node, depth=0):
        """
        used to set the root_node property for this node and all ROOT or
        TREE descendents
        """
        if self.class_ == WunschNode.TREE or self.class_ == WunschNode.ROOT:
            self.root_node = root_node
            self.tree_depth = depth
            for child in self.children:
                child.propagate_root(root_node, depth=depth+1)

    @property
    def is_in_tree(self):
        return self.class_ == WunschNode.ROOT or self.class_ == WunschNode.TREE

    @property
    def distance_to_inner(self):
        if self.is_in_tree:
            return self.tree_depth + 1
        elif self.class_ == WunschNode.LEAF:
            return 1
        else:
            return 0

    @property
    def inner_node(self):
        if self.is_in_tree:
            return self.root_node.get_single_parent()
        elif self.class_ == WunschNode.LEAF:
            return self.get_single_parent()
        else:
            return self

    def tree_distance(self, other):
        self_anc = self
        other_anc = other
        while (self_anc.tree_depth >= 0 and other_anc.tree_depth >= 0):
            if self_anc == other_anc:
                return (self.tree_depth - self_anc.tree_depth) + (other.tree_depth - other_anc.tree_depth)
            elif self_anc.tree_depth > other_anc.tree_depth:
                self_anc = self_anc.get_single_parent()
            else:
                other_anc = other_anc.get_single_parent()
        raise Exception('tree_distance not defined for nodes in separate trees: %s, %s' % (self, other))

class WunschPaths(object):
    """
    Algorithm from ``Exploiting graph structure for accelerating the calculation of shortest paths in Wordnets'',
    by Holger Wunsch
    """
    def __init__(self, synsets, cache_lookups=False):
        self.base_node = WunschNode(0, None)
        self.base_node.class_ = WunschNode.INNER
        self.nodes = [self.base_node] + [WunschNode(index, synset) for index, synset in enumerate(synsets, 1)]
        self.N_nodes = len(self.nodes)
        self.synset_to_node = dict((node.synset, node)
                                   for node in self.nodes)
        self.index_to_node = dict((node.index, node) for node in self.nodes)

        self.make_node_links()
        self.classify_nodes()

        self.build_inner_graph()

        self.cache_lookups = cache_lookups
        if self.cache_lookups:
            self.lookup_cache = {}

    def synset_path_distance(self, synset_1, synset_2):
        if self.cache_lookups:
            key_1 = (synset_1, synset_2)
            key_2 = (synset_2, synset_1)
            if key_1 in self.lookup_cache:
                return self.lookup_cache[key_1]
            if key_2 in self.lookup_cache:
                return self.lookup_cache[key_2]
        distance = self.node_path_distance(self.synset_to_node[synset_1], self.synset_to_node[synset_2])
        if self.cache_lookups:
            self.lookup_cache[(synset_1, synset_2)] = distance
        return distance

    def node_path_distance(self, x, y):
        if x.is_in_tree and y.is_in_tree and x.root_node == y.root_node:
            # x and y belong to the same tree
            return x.tree_distance(y)
        else:
            # x and y don't belong to the same tree
            return x.distance_to_inner + y.distance_to_inner + self.inner_shortest_path_lengths[x.inner_node.inner_index, y.inner_node.inner_index]

    def make_node_links(self):
        for node in self.nodes:
            if not node.synset:
                # the base node
                continue
            hypernyms = node.synset.hypernyms()
            for hypernym in hypernyms:
                hypernym_node = self.synset_to_node[hypernym]
                node.parents.add(hypernym_node)
                hypernym_node.children.add(node)
            if not hypernyms:
                # connect to base node
                self.base_node.children.add(node)
                node.parents.add(self.base_node)

    def classify_nodes(self):
        print "classifying nodes"
        # classify leaf nodes
        for node in self.nodes:
            if not node.children and len(node.parents) == 1:
                node.class_ = WunschNode.LEAF
        # classify inner nodes
        for node in self.nodes:
            if len(node.parents) > 1:
                if node.class_ != WunschNode.INNER:
                    node.class_ = WunschNode.INNER
                    for ancestor in node.ancestors():
                        ancestor.class_ = WunschNode.INNER
        # classify root nodes and trees
        for node in self.nodes:
            if node.class_ is None:
                # should only be one parent, otherwise would have been
                # classified as INNER in previous step
                if node.get_single_parent().class_ == WunschNode.INNER:
                    node.class_ = WunschNode.ROOT
                else:
                    node.class_ = WunschNode.TREE
        # reclassify leaf nodes as tree nodes if they're children of tree nodes
        for node in self.nodes:
            if node.class_ == WunschNode.LEAF:
                parent = node.get_single_parent()
                # note: there appears to be an error in the pseudocode in the
                # paper, it says that nodes with ROOT or LEAF parents should be changed to
                # tree, but it's not possible for a LEAF node to have another
                # LEAF node as parent (since LEAF nodes have no children)
                if parent.class_ == WunschNode.ROOT or parent.class_ == WunschNode.TREE:
                    node.class_ = WunschNode.TREE

        # store a reference to the root node in all nodes of a tree
        for node in self.nodes:
            if node.class_ == WunschNode.ROOT:
                node.propagate_root(node)

    def build_inner_graph(self):
        inner_nodes = [node for node in self.nodes if node.class_ == WunschNode.INNER]
        self.N_inner_nodes = len(inner_nodes)
        self.inner_index_to_node = dict(enumerate(inner_nodes))
        for inner_index, node in self.inner_index_to_node.items():
            node.inner_index = inner_index
        ## Floyd-Warshall on all INNER nodes
        print 'starting Floyd Warshall'
        print 'creating array for %d nodes' % self.N_inner_nodes

        P = np.matrix(np.empty((self.N_inner_nodes, self.N_inner_nodes)))
        # initialize node weights
        print 'initializing node weights'
        P.fill(np.inf)
        P[np.identity(self.N_inner_nodes) == 1] = 0
        for node in inner_nodes:
            for child in node.children:
                if child.class_ == WunschNode.INNER:
                    P[node.inner_index, child.inner_index] = 1
                    P[child.inner_index, node.inner_index] = 1
        for z in range(self.N_inner_nodes):
            sys.stdout.write('Floyd-Warshall %d / %d (%0.2f)\r' % (z, self.N_inner_nodes, float(z) / self.N_inner_nodes))
            sys.stdout.flush()
            P = np.minimum(P, P[z,:] + P[:,z])
            # if z % 10 == 0:
            #     print P
        print
        self.inner_shortest_path_lengths = P

    def build_all_pairs(self):
        rows = []
        N = len(self.nodes)
        for index, x in enumerate(self.nodes):
            sys.stdout.write('all pairs row %d / %d (%f) %s\r' %  (index, N, float(index) / N, x))
            rows.append(np.fromiter((self.path_distance(x, y) for y in self.nodes), dtype=np.int))
        print
        self.all_pairs_shortest_path_lengths = np.matrix(rows)

def safe_similarity_wrapper(similarity_fn):
    def foo(synset1, synset2):
        try:
            return similarity_fn(synset1, synset2)
        except:
            # print synset1, synset2
            return None
    return foo

def ancestor_jaccard(synset1, synset2):
    word1_ancestors = set(s for lst in synset1.hypernym_paths() for s in lst)
    word1_ancestors.add(synset1)
    word2_ancestors = set(s for lst in synset2.hypernym_paths() for s in lst)
    word2_ancestors.add(synset2)
    return len(word1_ancestors.intersection(word2_ancestors)) / float(len(word1_ancestors.union(word2_ancestors)))

def inspect_best(word1, word2, similarity_fn=safe_similarity_wrapper(wn.lch_similarity), n=5):
    pairs = list(itertools.product(wn.synsets(word1), wn.synsets(word2)))
    for score, (s1, s2) in sorted(zip((similarity_fn(s1, s2) for s1, s2 in pairs), pairs), reverse=True)[:n]:
        print '%s: <%s (%s)> -- <%s (%s)>' % (score, s1, s1.definition[:50], s2, s2.definition[:50])

def pairwise_similarity(synsets1, synsets2, similarity_fn=safe_similarity_wrapper(wn.lch_similarity), reduction_fn=np.max, remove_zeros=False):
    """
    Compute the simlarity between two lists of synsets, using a pairwise sense similarity function. Pairwise scores
    are calculated for all synsets, and then aggregated using the given reduction function (so to return the minimum
    or maximum similarity, pass in min or max. Some similarity functions will return None if two synsets are not
    comparable (e.g. there is no path between them in the ontology). remove_zeros determines if None and 0 values should
    be removed before the aggregation, or if only Nones should be removed

    @type synsets1: List of L{Synset}
    @param synsets1: First list of synsets. Each item in this list is compared to each item in synsets2

    @type synsets2: List of L{Synset}
    @param synsets2: First list of synsets. Each item in this list is compared to each item in synsets1

    @type similarity_fn: Function of two L{Synset}s
    @param similarity_fn: Function to be mapped over each pair of synsets

    @type reduction_fn: Function of sequence of reals
    @param reduction_fn: Reduces the pariwise similarities

    @type remove_zeros: bool
    @param remove_zeros: If true, remove None and 0 similarity values. If false, replace None with 0 and include
    """
    pairwise_sims = (similarity_fn(synset1, synset2) for (synset1, synset2) in itertools.product(synsets1, synsets2))
    if remove_zeros:
        replaced = [s for s in pairwise_sims if (s is not None and s != 0)]
    else:
        replaced = [s for s in pairwise_sims if s is not None]
    if not replaced:
        return None
    else:
        return reduction_fn(replaced)

def make_similarity_matrix(words, print_output=True, **kwargs):
    # todo: how do we handle the diagonals?
    N = len(words)
    print 'getting synsets'
    synsets_for_word = [wn.synsets(word) for word in words]
    similarity_matrix = np.zeros((N, N))
    for index1 in xrange(N):
        word1 = words[index1]
        synsets1 = synsets_for_word[index1]
        if print_output:
            sys.stdout.write('\r%i / %i \t (%0.2f) \t %s' % (index1, N, float(index1) / N, word1))
            sys.stdout.flush()
        vals = np.array([pairwise_similarity(synsets1, synsets_for_word[index2], **kwargs)
                for index2 in xrange(index1, N)])
        similarity_matrix[index1,index1:] = vals
        similarity_matrix[index1:,index1] = vals
    print
    return similarity_matrix
