from nltk.corpus import wordnet as wn
import itertools
import numpy as np
import sys

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
    def __init__(self, synsets):
        self.base_node = WunschNode(0, None)
        self.nodes = [self.base_node] + [WunschNode(index, synset) for index, synset in enumerate(synsets, 1)]
        self.N_nodes = len(self.nodes)
        self.synset_to_node = dict((node.synset, node)
                                   for node in self.nodes)
        self.index_to_node = dict((node.index, node) for node in self.nodes)

        self.make_node_links()
        self.classify_nodes()

        self.build_inner_graph()

    def path_distance(self, x, y):
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
                if parent.class_ == WunschNode.ROOT or parent.class_ == WunschNode.LEAF:
                    node.class_ == WunschNode.TREE

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

def safe_similarity_wrapper(similarity_fn):
    def foo(synset1, synset2):
        try:
            return similarity_fn(synset1, synset2)
        except:
            # print synset1, synset2
            return None
    return foo

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
    be removed before the aggregation, or if Nones should be replaced with 0s

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
        replaced = [s if s is not None else 0 for s in pairwise_sims]
    else:
        replaced = [s for s in pairwise_sims if s is not None]
    if not replaced:
        replaced = [0]
    return reduction_fn(replaced)

def make_similarity_matrix(words, top_k=None, print_output=True, **kwargs):
    # todo: how do we handle the diagonals?
    N = len(words)
    similarity_matrix = np.zeros((top_k if top_k else N, N))
    no_synsets_for = []
    for (index1, word1) in enumerate(words):
        if top_k and index1 >= top_k:
            break
        synsets1 = wn.synsets(word1)
        if print_output:
            max_ = top_k if top_k else N
            sys.stdout.write('\r%i / %i \t (%0.2f) \t %s' % (index1, max_, float(index1) / max_, word1))
            sys.stdout.flush()
        if not synsets1:
            no_synsets_for.append(word1)
            continue
        for (index2, word2) in enumerate(words):
            # if index2 == index1:
            #     continue
            synsets2 = wn.synsets(word2)
            v = pairwise_similarity(synsets1, synsets2, **kwargs)
            similarity_matrix[index1, index2] = v
            if top_k and index2 < top_k:
                similarity_matrix[index2, index1] = v
    print
    print 'no synsets for %i words' % len(no_synsets_for)
    return similarity_matrix
