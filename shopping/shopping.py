import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


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
    evidence = []
    labels = []
    months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "june": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    with open(filename, "r") as file:
        file.readline()  # ignore header
        temp = file.readline()
        while temp != "":
            temp_evidence = []
            splitted = temp.split(",")

            # Convert specific fields to appropriate data types
            temp_evidence.append(int(splitted[0]))  # Administrative, an integer
            temp_evidence.append(
                float(splitted[1])
            )  # Administrative_Duration, a floating point number
            temp_evidence.append(int(splitted[2]))  # Informational, an integer
            temp_evidence.append(
                float(splitted[3])
            )  # Informational_Duration, a floating point number
            temp_evidence.append(int(splitted[4]))  # ProductRelated, an integer
            temp_evidence.append(
                float(splitted[5])
            )  # ProductRelated_Duration, a floating point number
            temp_evidence.append(
                float(splitted[6])
            )  # BounceRates, a floating point number
            temp_evidence.append(
                float(splitted[7])
            )  # ExitRates, a floating point number
            temp_evidence.append(
                float(splitted[8])
            )  # PageValues, a floating point number
            temp_evidence.append(
                float(splitted[9])
            )  # SpecialDay, a floating point number
            temp_evidence.append(
                months[splitted[10].lower().strip()] - 1
            )  # Month, an index from 0 to 11
            temp_evidence.append(int(splitted[11]))  # OperatingSystems, an integer
            temp_evidence.append(int(splitted[12]))  # Browser, an integer
            temp_evidence.append(int(splitted[13]))  # Region, an integer
            temp_evidence.append(int(splitted[14]))  # TrafficType, an integer
            temp_evidence.append(
                int(splitted[15].strip() == "Returning_Visitor")
            )  # VisitorType, an integer 0 or 1
            temp_evidence.append(
                int(splitted[16].strip() == "TRUE")
            )  # Weekend, an integer 0 or 1

            labels.append(
                int(splitted[-1].strip() == "TRUE")
            )  # Revenue is true (1) or false (0)

            evidence.append(temp_evidence)
            temp = file.readline()
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    X_train, _, y_train, _ = train_test_split(evidence, labels)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    return knn


def evaluate(labels, predictions):
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
    true_pos = sum(
        (label == 1 and pred == 1) for label, pred in zip(labels, predictions)
    )
    true_neg = sum(
        (label == 0 and pred == 0) for label, pred in zip(labels, predictions)
    )
    false_pos = sum(
        (label == 0 and pred == 1) for label, pred in zip(labels, predictions)
    )
    false_neg = sum(
        (label == 1 and pred == 0) for label, pred in zip(labels, predictions)
    )
    sensitivity = true_pos / (true_pos + false_neg)
    specificity = true_neg / (true_neg + false_pos)
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
