import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    pagesToGo = corpus[page]

    # Initialize the output dictionary with all pages in the corpus
    out = {p: 0 for p in corpus.keys()}

    num_links = len(pagesToGo)
    for linked_page in pagesToGo:
        out[linked_page] = damping_factor / num_links

    for p in corpus.keys():
        out[p] += (1 - damping_factor) / len(corpus.keys())

    return out


import random


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random[^1^][1].

    Return a dictionary where keys are page names, and values are[^2^][2]
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize the visits dictionary
    visits = {page: 0 for page in corpus.keys()}

    # Choose the first page at random
    pages = list(corpus.keys())
    page = random.choice(pages)

    for _ in range(n):
        # Update the visits count
        visits[page] += 1

        # Get the transition model for the current page
        prob = transition_model(corpus, page, damping_factor)

        # Choose the next page based on the transition probabilities
        page = random.choices(pages, [prob[page] for page in pages])[0]

    # Calculate the estimated PageRank for each page
    out = {page: visits[page] / n for page in pages}

    return out


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    ranks = {page: 1 / len(corpus) for page in corpus.keys()}

    while True:
        rCopy = ranks.copy()

        for page in ranks.keys():
            ranks[page] = (1 - damping_factor) / len(corpus)
            sum = 0
            for p in corpus.keys():
                if corpus[p]:
                    links = len(corpus[p])
                    if page in corpus[p]:
                        sum += rCopy[p] / links
                else:
                    sum += rCopy[p] / len(corpus)

            ranks[page] += damping_factor * sum

        if checkConvergence(ranks, rCopy):
            break
    return ranks


def checkConvergence(ranks, rCopy):
    for page in ranks.keys():
        if abs(ranks[page] - rCopy[page]) > 0.001:
            return False
    return True


if __name__ == "__main__":
    main()
