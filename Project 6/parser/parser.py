import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP_Adv VP_Adv | S Conj S
NP -> N | Det N | Det Adj_Stack N | NP Conj NP | NP P NP
VP -> V | V NP | V P NP | VP Conj VP
Adj_Stack -> Adj | Adj Adj_Stack
NP_Adv -> NP | NP Adv
VP_Adv -> VP | VP Adv | Adv VP

"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence: str):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    sentence = sentence.lower()
    words = []
    for word in nltk.tokenize.word_tokenize(sentence):
        # If no aphabetic character exists in word, ignore it
        if not any(char.isalpha() for char in word):
            continue
        words.append(word)
    return words
    raise NotImplementedError


def np_chunk_rec(tree: nltk.Tree, np: list[nltk.Tree]):
    # The base case is implicit as the for loop ignores any subtrees with type != "nltk.Tree"
    for subtree in tree:
        if type(subtree) == nltk.Tree:
            if subtree.label() == "NP":
                nested = False
                # If any tree "s" from the tree "subtree" has a label "NP", then 
                # "subtree" contains other noun phrases in it, Ignore that "subtree"
                for s in subtree.subtrees():
                    if s != subtree and s.label() == "NP":
                        nested = True
                        break
                if not nested:
                    np.append(subtree)
            np_chunk_rec(subtree, np)


def np_chunk(tree: nltk.Tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    np = []
    # I decided to use a recursive function as I am more comfortable with recursive tree traversal
    np_chunk_rec(tree, np)
    return np
    raise NotImplementedError


if __name__ == "__main__":
    main()
