import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

# Helper lists for CSV Formatting
INTEGER_HEADERS = [
    "Administrative",
    "Informational",
    "ProductRelated",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]
FLOAT_HEADERS = [
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]
MONTHS = {
    "Jan": 0,
    "Feb": 1,
    "Mar": 2,
    "Apr": 3,
    "May": 4,
    "June": 5,
    "Jul": 6,
    "Aug": 7,
    "Sep": 8,
    "Oct": 9,
    "Nov": 10,
    "Dec": 11,
}

VISITOR = {
    "Returning_Visitor": 1,
    "New_Visitor": 0,
    "Other": 0,
}

WEEKEND = {"FALSE": 0, "TRUE": 1}

LABELS = {"FALSE": 0, "TRUE": 1}


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = list()
    labels = list()

    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence_entry = list()
            items = list(row.items())
            # Iterate over all items in row except the last one
            for key, val in items[:-1]:
                if key in INTEGER_HEADERS:
                    evidence_entry.append(int(val))
                elif key in FLOAT_HEADERS:
                    evidence_entry.append(float(val))
                elif key == "Month":
                    evidence_entry.append(MONTHS[val])
                elif key == "VisitorType":
                    evidence_entry.append(VISITOR[val])
                elif key == "Weekend":
                    evidence_entry.append(WEEKEND[val])
            # Add the new evidence entry and label to the evidence/labels list
            evidence.append(evidence_entry)
            labels.append(LABELS[row["Revenue"]])
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    return model.fit(evidence, labels)


def evaluate(labels: list, predictions: list):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positive = 0  # Number of cases where the prediction is correctly 1
    true_negative = 0  # Number of cases where the prediction is correctly 0

    # Get the number of true_positive and true_negative cases in labels and predictions
    for label, prediction in zip(labels, predictions):
        if label == prediction and prediction == 1:
            true_positive += 1
        if label == prediction and prediction == 0:
            true_negative += 1
    all_positive = labels.count(1)
    all_negative = labels.count(0)
    return (true_positive / all_positive, true_negative / all_negative)


if __name__ == "__main__":
    main()
