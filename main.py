import argparse
from typing import Tuple
from numpy import ndarray, ravel
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def read_args() -> Tuple[int, int]:
    """
    Function that reads the command line arguments.

    Returns:
    - Tuple[int, int]: starting_seed, num_runs
    """

    # Create argument parser
    parser = argparse.ArgumentParser(description="Process command line arguments.")

    # Add flags for starting seed and number of runs
    parser.add_argument(
        "--seed", type=int, default=0, help="Starting seed for random number generation"
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")

    # Parse command line arguments
    args = parser.parse_args()

    # Access the values of the flags
    starting_seed = args.seed
    num_runs = args.runs

    return starting_seed, num_runs


def load_data() -> Tuple[DataFrame, DataFrame]:
    """
    Function that loads the Titanic dataset.

    Returns:
    - Tuple[DataFrame, DataFrame]: train_data, test_data
    """
    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")

    return train_data, test_data


def preprocess_data(original_data: DataFrame) -> Tuple[ndarray, ndarray]:
    """
    Function that preprocesses the data for the Titanic dataset.

    Args:
    - data: DataFrame

    Returns:
    - Tuple[ndarray, ndarray]: x, y
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
    data = original_data.copy(deep=True)
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
        y = DataFrame({"Survived": []})

    # --- CATEGORIZE ENCODING AND AVOID DUMMY VARIABLES TRAP ---
    x = pd.get_dummies(x, drop_first=True)

    return x.values, ravel(y.values)


def solve(
    usable_dataframe: DataFrame, competition_dataframe: DataFrame, random_state: int = 0
) -> Tuple[float, ndarray, DataFrame]:
    """
    A function that runs the entire pipeline for the Titanic dataset.
    """

    # --- PREPROCESSING ---
    x_usable, y_usable = preprocess_data(usable_dataframe)
    x_competition, y_competition = preprocess_data(competition_dataframe)

    # --- SPLIT ---
    x_train, x_test, y_train, y_test = train_test_split(
        x_usable,
        y_usable,
        test_size=0.25,
        random_state=random_state,
    )
    x_train, y_train = x_usable, y_usable

    # --- SUPERVISE ---
    model = RandomForestClassifier(n_estimators=69, criterion="entropy", random_state=0)
    model.fit(x_train, y_train)

    # --- TEST_MODEL ---
    y_test_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_test_pred)
    acs = accuracy_score(y_test, y_test_pred)

    # print(f"Your confusion matrix:\n{cm}")
    # print(f"Your accuracy score (%):\n{round(acs * 100, 2)}")

    # --- PREDICT_COMPETITION ---
    y_competition_pred = model.predict(x_competition)
    output = DataFrame(
        {
            "PassengerId": competition_dataframe["PassengerId"],
            "Survived": y_competition_pred,
        }
    )

    return float(acs), cm, output


if __name__ == "__main__":
    # --- LOAD COMMAND LINE FLAGS ---
    starting_seed, num_runs = read_args()

    # --- LOAD DATA ---
    usable_dataframe, competition_dataframe = load_data()

    # --- SOLVE ---

    # TODO: Implement a correct k-fold cross validation
    max_score: float = 0.0
    max_solution: DataFrame = DataFrame()
    max_cm: ndarray = ndarray([])
    max_seed: int = 0

    for seed in range(starting_seed, starting_seed + num_runs):
        score, cm, solution = solve(
            usable_dataframe=usable_dataframe,
            competition_dataframe=competition_dataframe,
            random_state=seed,
        )
        if score > max_score:
            max_score = score
            max_solution = solution
            max_cm = cm
            max_seed = seed

    print(f"Max score: {max_score} | Max seed: {max_seed}")
    print(f"Max confusion matrix:\n{max_cm}")
    max_solution.to_csv(
        f"submission_{round(max_score, 5)}_{max_cm[0][0]}-{max_cm[0][1]}-{max_cm[1][0]}-{max_cm[1][1]}_{max_seed}.csv",
        index=False,
    )
    print("Your max score solution was successfully saved!")
