from __future__ import print_function, unicode_literals
from collections import defaultdict
from pprint import pformat
from nltk.compat import python_2_unicode_compatible, string_types
import nltk

@python_2_unicode_compatible
class DependencyGraph(object):

    @staticmethod
    def from_sentence(sent):
        tokens = nltk.word_tokenize(sent)
        tagged = nltk.pos_tag(tokens)
        
        dg = DependencyGraph()
        for (index, (word, tag)) in enumerate(tagged):
            dg.nodes[index + 1] = {
                'word': word,
                'lemma': '_',
                'ctag': tag,
                'tag': tag,
                'feats': '_',
                'rel': '_',
                'deps': defaultdict(),
                'head': '_',
                'address': index + 1,
            }
        
        return dg
        
    """
    A container for the nodes and labelled edges of a dependency structure.
    """

    def __init__(self, tree_str=None, cell_extractor=None, zero_based=False, cell_separator=None):
        """Dependency graph.

        We place a dummy `TOP` node with the index 0, since the root node is
        often assigned 0 as its head. This also means that the indexing of the
        nodes corresponds directly to the Malt-TAB format, which starts at 1.

        If zero-based is True, then Malt-TAB-like input with node numbers
        starting at 0 and the root node assigned -1 (as produced by, e.g.,
        zpar).

        :param str cell_separator: the cell separator. If not provided, cells
        are split by whitespace.

        """
        self.nodes = defaultdict(lambda: {'deps': defaultdict(list)})
        self.nodes[0].update(
            {
                'word': None,
                'lemma': None,
                'ctag': 'TOP',
                'tag': 'TOP',
                'feats': None,
                'rel': 'TOP',
                'address': 0,
            }
        )

        self.root = None

        if tree_str:
            self._parse(
                tree_str,
                cell_extractor=cell_extractor,
                zero_based=zero_based,
                cell_separator=cell_separator,
            )

    def remove_by_address(self, address):
        """
        Removes the node with the given address.  References
        to this node in others will still exist.
        """
        del self.nodes[address]

    def redirect_arcs(self, originals, redirect):
        """
        Redirects arcs to any of the nodes in the originals list
        to the redirect node address.
        """
        for node in self.nodes.values():
            new_deps = []
            for dep in node['deps']:
                if dep in originals:
                    new_deps.append(redirect)
                else:
                    new_deps.append(dep)
            node['deps'] = new_deps

    def add_arc(self, head_address, mod_address):
        """
        Adds an arc from the node specified by head_address to the
        node specified by the mod address.
        """
        relation = self.nodes[mod_address]['rel']
        self.nodes[head_address]['deps'].setdefault(relation,[])
        self.nodes[head_address]['deps'][relation].append(mod_address)


    def get_by_address(self, node_address):
        """Return the node with the given address."""
        return self.nodes[node_address]

    def contains_address(self, node_address):
        """
        Returns true if the graph contains a node with the given node
        address, false otherwise.
        """
        return node_address in self.nodes

    def __str__(self):
        return pformat(self.nodes)

    def __repr__(self):
        return "<DependencyGraph with {0} nodes>".format(len(self.nodes))

    @staticmethod
    def load(filename, zero_based=False, cell_separator=None):
        """
        :param filename: a name of a file in Malt-TAB format
        :param zero_based: nodes in the input file are numbered starting from 0
        rather than 1 (as produced by, e.g., zpar)
        :param str cell_separator: the cell separator. If not provided, cells
        are split by whitespace.

        :return: a list of DependencyGraphs

        """
        with open(filename) as infile:
            return [
                DependencyGraph(
                    tree_str,
                    zero_based=zero_based,
                    cell_separator=cell_separator,
                )
                for tree_str in infile.read().split('\n\n')
            ]

    def left_children(self, node_index):
        """
        Returns the number of left children under the node specified
        by the given address.
        """
        children = self.nodes[node_index]['deps']
        index = self.nodes[node_index]['address']
        return sum(len([i for i in children[rel] if i < index]) for rel in children if rel!='_')
        #return sum(1 for c in children if c < index)

    def right_children(self, node_index):
        """
        Returns the number of right children under the node specified
        by the given address.
        """
        children = self.nodes[node_index]['deps']
        index = self.nodes[node_index]['address']
        return sum(len([i for i in children[rel] if i > index]) for rel in children if rel!='_')
        #return sum(1 for c in children if c > index)

    def add_node(self, node):
        if not self.contains_address(node['address']):
            self.nodes[node['address']].update(node)

    def _parse(self, input_, cell_extractor=None, zero_based=False, cell_separator=None):
        """Parse a sentence.

        :param extractor: a function that given a tuple of cells returns a
        7-tuple, where the values are ``word, lemma, ctag, tag, feats, head,
        rel``.

        :param str cell_separator: the cell separator. If not provided, cells
        are split by whitespace.

        """

        def extract_3_cells(cells):
            word, tag, head = cells
            return word, word, tag, tag, '', head, ''

        def extract_4_cells(cells):
            word, tag, head, rel = cells
            return word, word, tag, tag, '', head, rel

        def extract_10_cells(cells):
            _, word, lemma, ctag, tag, feats, head, rel, _, _ = cells
            return word, lemma, ctag, tag, feats, head, rel

        extractors = {
            3: extract_3_cells,
            4: extract_4_cells,
            10: extract_10_cells,
        }

        if isinstance(input_, string_types):
            input_ = (line for line in input_.split('\n'))

        lines = (l.rstrip() for l in input_)
        lines = (l for l in lines if l)

        for index, line in enumerate(lines, start=1):
            cells = line.split(cell_separator)
            nrCells = len(cells)

            if cell_extractor is None:
                try:
                    cell_extractor = extractors[nrCells]
                except KeyError:
                    raise ValueError(
                        'Number of tab-delimited fields ({0}) not supported by '
                        'CoNLL(10) or Malt-Tab(4) format'.format(nrCells)
                    )

            word, lemma, ctag, tag, feats, head, rel = cell_extractor(cells)

            head = int(head)
            if zero_based:
                head += 1

            self.nodes[index].update(
                {
                    'address': index,
                    'word': word,
                    'lemma': lemma,
                    'ctag': ctag,
                    'tag': tag,
                    'feats': feats,
                    'head': head,
                    'rel': rel,
                }
            )

            self.nodes[head]['deps'][rel].append(index)

        if not self.nodes[0]['deps']['ROOT']:
            raise DependencyGraphError(
                "The graph does'n contain a node "
                "that depends on the root element."
            )
        root_address = self.nodes[0]['deps']['ROOT'][0]
        self.root = self.nodes[root_address]


    def to_conll(self, style):
        """
        The dependency graph in CoNLL format.

        :param style: the style to use for the format (3, 4, 10 columns)
        :type style: int
        :rtype: str
        """

        if style == 3:
            template = '{word}\t{tag}\t{head}\n'
        elif style == 4:
            template = '{word}\t{tag}\t{head}\t{rel}\n'
        elif style == 10:
            template = '{i}\t{word}\t{lemma}\t{ctag}\t{tag}\t{feats}\t{head}\t{rel}\t_\t_\n'
        else:
            raise ValueError(
                'Number of tab-delimited fields ({0}) not supported by '
                'CoNLL(10) or Malt-Tab(4) format'.format(style)
            )

        return ''.join(template.format(i=i, **node) for i, node in sorted(self.nodes.items()) if node['tag'] != 'TOP')



class DependencyGraphError(Exception):
    """Dependency graph exception."""
