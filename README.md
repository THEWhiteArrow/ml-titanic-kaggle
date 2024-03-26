# Machine Learning Project: Titanic Survival Prediction

This is a beginner-friendly machine learning project where we use Random Forest Classification to predict who survived the Titanic catastrophe. The project is based on the Kaggle Titanic competition and serves as a good starting point for beginners in the field of machine learning.

## Getting Started

#### Default

In order to run this project, you need to have Python installed on your machine. Install the dependencies using the `requirements.txt` file with `pip install -r requirements.txt`.
Download the dataset from the [Kaggle competition page](https://www.kaggle.com/c/titanic/data) and extract it into the `data` directory. Please note that due to the rules of the challenge/competition, the data files should not be shared with individuals who have not agreed to the competition rules.
In order to run the project navigate to the main directory and run the main file using the command `python main.py`.

#### Advanced

The project utlizes random seed for reproducibility. By default it is set to `0` and runs the model only once. If you can change this behavior by running `main.py` with the following arguments:

- `--seed` - set the random seed for reproducibility
- `--runs` - set the number of runs for the model to run
  The seed will be the sum of the number of the run and the seed. For example, if you run the model with `--seed 0 --runs 5`, the seeds will be `0, 1, 2, 3, 4`.
  Example: `python main.py --seed 0 --runs 5`

## Analysis

The project will generate a submission file in the root directory of the project. The file will contain the predictions of the model on the test dataset. The file name will have format `submission_{accuracy}_{true_negatives}_{false_positives}_{false_negatives}_{true_positives}_{seed}.csv`.
Legend:

- `accuracy` - the accuracy of the model
- `true_negatives` - the number of true negatives
- `false_positives` - the number of false positives
- `false_negatives` - the number of false negatives
- `true_positives` - the number of true positives
- `seed` - the seed used for the model

## Dataset

Dataset belongs to the Kaggle Titanic competition and can be found [here](https://www.kaggle.com/c/titanic/data).
