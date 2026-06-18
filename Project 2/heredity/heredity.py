import csv
import itertools
import sys

PROBS = {
    # Unconditional probabilities for having gene
    "gene": {2: 0.01, 1: 0.03, 0: 0.96},
    "trait": {
        # Probability of trait given two copies of gene
        2: {True: 0.65, False: 0.35},
        # Probability of trait given one copy of gene
        1: {True: 0.56, False: 0.44},
        # Probability of trait given no gene
        0: {True: 0.01, False: 0.99},
    },
    # Mutation probability
    "mutation": 0.01,
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {"gene": {2: 0, 1: 0, 0: 0}, "trait": {True: 0, False: 0}}
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (
                people[person]["trait"] is not None
                and people[person]["trait"] != (person in have_trait)
            )
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (
                    True
                    if row["trait"] == "1"
                    else False if row["trait"] == "0" else None
                ),
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s)
        for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def gene_count(person, one_gene, two_genes):
    if person in two_genes:
        return 2
    if person in one_gene:
        return 1
    return 0


def inheret_prob(gene_count):
    if gene_count == 0:
        return PROBS["mutation"]
    elif gene_count == 1:
        return 0.5
    else:
        return 1 - PROBS["mutation"]


def joint_helper(people, person, trait_flag, gene_num, one_gene, two_genes):
    """
    Helper for joint probability.

    params:
        trait_flag: a flag for checking if it is a trait(True), a trait(False)
                    0 -> trait(False)
                    1 -> trait(True)
        gene_num: The number of genes required for calculation
    """
    # Check if no parents
    if people[person]["mother"] is None and people[person]["father"] is None:
        res = PROBS["gene"][gene_num] * PROBS["trait"][gene_num][trait_flag]
        return res
    else:
        # We want to get the probability of having x genes based on probabilities of parents passing x genes
        # Probability that person has x genes
        gene_prob = 0
        mother = people[person]["mother"]
        father = people[person]["father"]

        # Gene count of paraents
        mother_genes = gene_count(mother, one_gene, two_genes)
        father_genes = gene_count(father, one_gene, two_genes)

        # Probability to inheret gene from parents
        mother_passes = inheret_prob(mother_genes)
        father_passes = inheret_prob(father_genes)
        # 3 cases:
        #   0 genes: (!mother & !father)
        if gene_num == 0:
            gene_prob = (1 - mother_passes) * (1 - father_passes)
        #   1 gene: (!mother & father) | (mother & !father)
        elif gene_num == 1:
            gene_prob = (mother_passes * (1 - father_passes)) + (
                (father_passes) * (1 - mother_passes)
            )
        #   2 genes: (mother & father)
        else:
            gene_prob = mother_passes * father_passes

        return gene_prob * PROBS["trait"][gene_num][trait_flag]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    joint_prob_res = 1

    # Calculate the joint probability be multiplying the individual results from joint_helper()
    for person in people:
        trait_flag = people[person]["name"] in have_trait
        gene_num = gene_count(person, one_gene, two_genes)
        joint_prob_res *= joint_helper(
            people, person, trait_flag, gene_num, one_gene, two_genes
        )

    return joint_prob_res

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # for each person in probabilities
    for person in probabilities:
        # Get the triat and gen_num probabilities we want to change
        trait_flag = person in have_trait
        gene_num = gene_count(person, one_gene, two_genes)

        probabilities[person]["gene"][gene_num] += p
        probabilities[person]["trait"][trait_flag] += p
    return


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        prob = 0
        for key in probabilities[person]:
            prob = 0
            # add the probabilites up
            for subkey in probabilities[person][key]:
                prob += probabilities[person][key][subkey]
            for subkey in probabilities[person][key]:
                probabilities[person][key][subkey] /= prob
    return


if __name__ == "__main__":
    main()
