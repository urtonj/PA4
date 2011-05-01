import re

from nltk.corpus.reader import *
from nltk.tag.util import str2tuple
from nltk.tokenize import RegexpTokenizer
from nltk.tree import Tree

def superchunk2tree(s, chunk_node="NP", top_node="S", sep='/'):
    """
    Divide a string of bracketted tagged text into
    chunks and unchunked tokens, and produce a C{Tree}.
    Chunks are marked by square brackets (C{[...]}).  Words are
    delimited by whitespace, and each word should have the form
    C{I{text}/I{tag}}.  Words that do not contain a slash are
    assigned a C{tag} of C{None}.

    @return: A tree corresponding to the string representation.
    @rtype: C{tree}
    @param s: The string to be converted
    @type s: C{string}
    @param chunk_node: The label to use for chunk nodes
    @type chunk_node: C{string}
    @param top_node: The label to use for the root of the tree
    @type top_node: C{string}
    """

    WORD_OR_BRACKET = re.compile(r'\[|\]|[^\[\]\s]+')

    stack = [Tree(top_node, [])]
    for match in WORD_OR_BRACKET.finditer(s):
        text = match.group()
        if text[0] == '[':
            chunk = Tree(chunk_node, [])
            stack[-1].append(chunk)
            stack.append(chunk)
        elif text[0] == ']':
            stack.pop()
        else:
            if sep is None:
                stack[-1].append(text)
            else:
                t = str2tuple(text, sep)
                if t[1] is None:
                    # Chunk label.
                    stack[-1].node = t[0]
                else:
                    stack[-1].append(t)

    if len(stack) != 1:
        raise ValueError('Expected ] at char %d' % len(s))
    return stack[0]

treebank_superchunk = ChunkedCorpusReader('.', r'wsj_.*\.pos',
    sent_tokenizer=RegexpTokenizer(r'(?<=/\.)\s*(?![^\[]*\])', gaps=True),
    para_block_reader=tagged_treebank_para_block_reader,
    str2chunktree=superchunk2tree)

def tree2iob(x, prefix="O", label="", super_prefix="O", super_label="",
             issuperchunk=lambda tree: tree.node=='SNP',
             issentence=lambda tree: tree.node=='S'):
    """Given a tree containing chunks and superchunks, yield tuples of the
    form (word, POS-tag, chunk-IOB-tag, superchunk-IOB-tag)."""
    if isinstance(x, Tree):
        if issuperchunk(x):
            super_prefix = "B-"
            super_label = x.node
        elif not issentence(x):
            prefix = "B-"
            label = x.node
        for child in x:
            for tag in tree2iob(child, prefix, label,
                                super_prefix, super_label,
                                issuperchunk, issentence):
                yield tag
            if prefix == "B-": prefix = "I-"
            if super_prefix == "B-": super_prefix = "I-"
    else:
        yield (x[0], x[1], prefix+label, super_prefix+super_label)
