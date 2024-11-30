# %%
import sys
import os
from enum import Enum
import numpy as np
import csv
import matplotlib.pyplot as plot


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Errors(Enum):
    """
    Enum class that inherits from :class:`Enum`
    """
    NO_ERRORS = 0
    INVALID_ARGUMENT = 1


def log_error(message: str) -> None:
    """
    Logs the provided error message to STDERR

    :param message: the error message to be logged
    :type message: str
    """
    print(f"{bcolors.FAIL}ERROR: {message}{bcolors.ENDC}", file=sys.stderr)


def read_csv(path: str) -> tuple[np.ndarray, list[str]]:
    """
    Reads the provided csv file and returns a numpy array with the read values
    and a list of strings representing the headers

    :param path: the path to the provided csv file
    :type path: str
    :returns: a tuple containing a numpy array and a list with the headers
    :rtype: tuple[np.ndarray, list[str]]
    """
    with open(path) as datafile:
        reader = csv.reader(datafile)
        datalist: list[list[float]] = [[], []]
        header: list[str] = next(reader)
        for row in reader:
            datalist[0].append(row[0])
            datalist[1].append(row[1])
        return (np.asarray(datalist, dtype=float), header)


def min_max_normalize(data: np.ndarray) -> np.ndarray:
    minimum1 = min(data[0])
    diff1 = max(data[0]) - minimum1
    minimum2 = min(data[1])
    diff2 = max(data[1]) - minimum2
    res1 = [((x - minimum1) / diff1) for x in data[0]]
    res2 = [((x - minimum2) / diff2) for x in data[1]]
    return np.asarray([res1, res2], dtype=float)


def linear_regression_train(
        data: np.ndarray[np.ndarray[float]],
        learning_rate: float = 0.01,
        epochs: int = 1000) -> None:
    """
    Trains the model with the data provided to this function

    :param data: 2D tensor. The data is expected to be divided in two
                sub-arrays
    :type data: np.ndarray
    :param alpha: the learning rate of the model, defaults to 0.01
    :type alpha: float
    :param epochs: the epochs (number of iterations) used by the model to
                get fine-tuned, defaults to 1000
    :type epochs: int
    """
    (x, y) = data
    m, b = .0, .0
    n = len(x)
    for _ in range(epochs):
        y_pred = m * x + b
        dm = (-2 / n) * np.sum(x * (y - y_pred))
        db = (-2 / n) * np.sum(y - y_pred)

        m -= learning_rate * dm
        b -= learning_rate * db

    plot.scatter(data[0], data[1])
    plot.plot([min(x), m * min(x) + b], [max(x), m * max(x) + b])
    plot.show()


def main():
    """
    The program accepts 1 argument which defines
    the path to the training data
    """
    if (len(sys.argv) != 2
            or not os.path.isfile(sys.argv[1])
            or not sys.argv[1].endswith(".csv")):
        log_error("Provide a valid path to a .csv file")
        exit(Errors.INVALID_ARGUMENT.value)
    (data, headers) = read_csv("./data/data.csv")
    norm_data = min_max_normalize(data)
    linear_regression_train(norm_data)


if __name__ == "__main__":
    main()

# %%
