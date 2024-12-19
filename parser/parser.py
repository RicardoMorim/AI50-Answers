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
S -> NP VP | S Conj S

NP -> N | Det N | Det AdjP N | NP PP | Det N PP | NP Adv | Det AdjP N PP
AdjP -> Adj | Adj AdjP
VP -> V | V NP | V NP PP | V PP | VP Adv | VP Conj VP
PP -> P NP | P S

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


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """

    def has_alpha_char(word):
        return any(c.isalpha() for c in word)

    sentence = sentence.lower()

    filtered_words = []

    for word in nltk.word_tokenize(sentence):
        if has_alpha_char(word):
            filtered_words.append(word)

    return filtered_words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """

    # Initialize an empty list to hold the noun phrase chunks
    np_chunks = []

    # Define a recursive function to check for 'NP' labels in the subtree
    def contains_np(subtree):
        # Check each child of the subtree
        for child in subtree:
            # If the child is a tree and has the label 'NP', return True
            if isinstance(child, nltk.tree.Tree):
                if child.label() == "NP":
                    return True
                # Recursively check the child's children
                elif contains_np(child):
                    return True
        # If no 'NP' labels are found in the subtree, return False
        return False

    # Define a recursive function to traverse the tree
    def traverse(t):
        # If the current subtree is 'NP'
        if t.label() == "NP":
            # If it does not contain any other 'NP', add it to the list
            if not contains_np(
                t[1:]
            ):  # Skip the first child, which is the current 'NP'
                np_chunks.append(t)
        # Traverse the children of the subtree
        for child in t:
            if isinstance(child, nltk.tree.Tree):
                traverse(child)

    # Start the traversal from the root of the tree
    traverse(tree)

    # Return the list of noun phrase chunks
    return np_chunks


if __name__ == "__main__":
    main()
