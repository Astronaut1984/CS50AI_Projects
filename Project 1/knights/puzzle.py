from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

AKnightOrKnave = And(Or(AKnight, AKnave), Not(And(AKnight, AKnave)))
BKnightOrKnave = And(Or(BKnight, BKnave), Not(And(BKnight, BKnave)))
CKnightOrKnave = And(Or(CKnight, CKnave), Not(And(CKnight, CKnave)))

# Puzzle 0
# A says "I am both a knight and a knave."
Sentence0 = And(AKnight, AKnave)
knowledge0 = And(
    AKnightOrKnave,
    Biconditional(AKnight, Sentence0),
    Implication(AKnave, Not(Sentence0))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
Sentence1 = And(AKnave, BKnave)
knowledge1 = And(
    AKnightOrKnave,
    BKnightOrKnave,
    Biconditional(AKnight, Sentence1),
    Implication(AKnave, Not(Sentence1))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
Sentence2a = Or(And(AKnight, BKnight), And(AKnave, BKnave))
Sentence2b = Or(And(AKnight, BKnave), And(AKnave, BKnight))
knowledge2 = And(
    AKnightOrKnave,
    BKnightOrKnave,
    Biconditional(AKnight, Sentence2a),
    Implication(AKnave, Not(Sentence2a)),
    Biconditional(BKnight, Sentence2b),
    Implication(BKnave, Not(Sentence2b)),
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
Sentence3a = Or(AKnight, AKnave)
Sentence3b = AKnave
Sentence3b2 = CKnave
Sentence3c = AKnight

knowledge3 = And(
    AKnightOrKnave,
    BKnightOrKnave,
    CKnightOrKnave,
    Biconditional(AKnight, Sentence3a),
    Implication(AKnave, Not(Sentence3a)),
    Biconditional(BKnight, And(Sentence3b, Sentence3b2)),
    Implication(BKnave, And(Not(Sentence3b), Not(Sentence3b2))),
    Biconditional(CKnight, Sentence3c),
    Implication(CKnave, Not(Sentence3c)),
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
