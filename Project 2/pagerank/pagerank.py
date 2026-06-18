import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000
TOLERANCE = 0.001


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    iterate_pagerank(corpus=corpus, damping_factor=DAMPING)
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
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    n = len(corpus)
    links = corpus[page]
    if len(links) == 0:
        res = dict.fromkeys(corpus, 1/n)
        return res
    else:
        # count the number of occurences
        res = dict.fromkeys(corpus, 0)
        for i in corpus[page]:
            res[i] += 1
        
        # divide by number of nodes
        for val in res.values():
            val /= n
        
        # calculate accounting for damping factor
        for key in res.keys():
            if key == page:
                res[key] = (1 - damping_factor) * (1/n)
            else:
                res[key] = res[key] * damping_factor + ((1-damping_factor) * (1/n))
        
        return res
    raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Initially, no clicks are made
    res = dict.fromkeys(corpus, 0)

    # https://stackoverflow.com/questions/40927221/how-to-choose-keys-from-a-python-dictionary-based-on-weighted-probability
    # Stack overflow ftw
    page = random.choices(list(corpus.keys()), k=1)[0]
    res[page] += 1
    prev = transition_model(corpus, page, damping_factor)

    # calculate how many times each page was clicked on
    for _ in range(1, n):
        # random.choices returns a list, so we get the first element from a 1 element 
        page = random.choices(list(prev.keys()), weights=prev.values(), k=1)[0]
        res[page] += 1
        prev = transition_model(corpus, page, damping_factor)
    
    # normalize each element
    for key in res.keys():
        res[key] /= n

    return res
    raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    n = len(corpus)
    pagerank = dict.fromkeys(corpus, 1/n)

    # Calculate the number of links coming out of each page
    numlinks = dict.fromkeys(corpus, 0)
    for key in corpus.keys():
        if len(corpus[key]) == 0:
            numlinks[key] = n
        else:
            numlinks[key] = len(corpus[key])

    above_thresh = True

    # Keep iterating while above threshold (tolerance)
    while above_thresh:
        new_pagerank = dict.fromkeys(corpus, 0)
        # Calculate the new pagerank for each page in the corpus
        for key in corpus.keys():
            addition = 0
            for i in corpus.keys():
                if key in corpus[i] or len(corpus[i]) == 0:
                    addition += pagerank[i] / numlinks[i]
            new_pagerank[key] = ((1-damping_factor)/n) + (damping_factor * addition)
        
        # Check if we are bellow threshold 
        above_thresh = False
        for page in corpus.keys():
            if abs(new_pagerank[page] - pagerank[page]) >= TOLERANCE:
                above_thresh = True
                break

        pagerank = copy.deepcopy(new_pagerank)
    
    return pagerank
    raise NotImplementedError


if __name__ == "__main__":
    main()
