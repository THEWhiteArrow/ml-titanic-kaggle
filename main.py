from typing import Tuple
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


def load_data() -> Tuple[DataFrame, DataFrame]:
    """
    Function that loads the Titanic dataset.

    Returns:
    - Tuple[DataFrame, DataFrame]: train_data, test_data
    """
    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")

    return train_data, test_data


def preprocess_data(data: DataFrame) -> Tuple[ndarray, ndarray | None]:
    """
    Function that preprocesses the data for the Titanic dataset.

    Args:
    - data: DataFrame

    Returns:
    - Tuple[ndarray, ndarray | None]: x, y
    """
    features_to_use = ["Pclass", "Sex", "SibSp", "Parch", "Age"]
    dependent_variable = ["Survived"]

    # --- MAP AND FILL NONES ---
    """
    Replace Cabin numbers with the first letter of the first cabin.
    This is done to reduce the number of unique values in the Cabin column.
    Provides the possible idea of where the cabin was located and if it was closer to the lifeboats.
    Additionally there is only one record of person in T cabin so lets simplify and remove it.
    """
    data["Cabin"] = (
        data["Cabin"]
        .fillna("Z")
        .map(lambda cabin: cabin[0] if cabin[0] != "T" else "Z")
    )

    def mapAge(age: float) -> str:
        """
        Maps the age to proper catogories as for life expectancy in 1912.

        Args:
        - age: float

        Returns:
        - str: "Child", "Adult", "Elderly"
        """
        if age < 18:
            return "Child"
        elif age < 50:
            return "Adult"
        else:
            return "Elderly"

    data["Age"] = data["Age"].fillna(data["Age"].mean()).map(mapAge)

    x = data[features_to_use]

    try:
        y = data[dependent_variable]
    except Exception:
        y = None

    # --- CATEGORIZE ENCODING AND AVOID DUMMY VARIABLES TRAP ---
    x = pd.get_dummies(x, drop_first=True)

    return x.values, y.values if y is not None else None


def main():
    """
    Main function that runs the entire pipeline for the Titanic dataset.
    """
    # --- SETUP ---
    train_dataframe, test_dataframe = load_data()

    # --- PREPROCESSING ---
    x_train, y_train = preprocess_data(train_dataframe)
    x_test, y_test = preprocess_data(test_dataframe)

    # --- SUPERVISE ---
    model = RandomForestClassifier(n_estimators=69, criterion="entropy", random_state=0)
    model.fit(x_train, y_train)

    # --- PREDICT_TEST ---
    y_test_pred = model.predict(x_test)
    output = pd.DataFrame(
        {"PassengerId": test_dataframe["PassengerId"], "Survived": y_test_pred}
    )

    output.to_csv("submission.csv", index=False)
    print("Your submission was successfully saved!")

    # --- DISPLAY ---
    """
    Training data is used since no test_data is provided to evaluate.
    """
    y_train_pred = model.predict(x_train)
    cm = confusion_matrix(y_train, y_train_pred)
    acs = accuracy_score(y_train, y_train_pred)
    print("Your confusion matrix:")
    print(cm)
    print("Your accuracy score (%):")
    print(round(acs * 100, 2))


if __name__ == "__main__":
    main()
