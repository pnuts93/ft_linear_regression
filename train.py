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
    (data, headers) = read_csv(sys.argv[1])
    plot.scatter(data[0], data[1])
    plot.show()
    print(data)


if __name__ == "__main__":
    main()
