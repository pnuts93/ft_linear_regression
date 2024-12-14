# %%
import sys
import json
from enum import Enum
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation


fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(14, 6))
theta: list[tuple[float, float]] = []
mse: list[float] = []
fig_data: np.ndarray
line, = ax1.plot([], [], color="red")
loss_line, = ax2.plot([], [], color="red")
min_norm_data: float
max_norm_data: float
min_x: float
max_x: float
min_y: float
max_y: float


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


def animate(i):
    line.set_data(
        [min_norm_data,
         max_norm_data],
        [theta[i][0] * min_norm_data + theta[i][1],
         theta[i][0] * max_norm_data + theta[i][1]])
    loss_line.set_data(range(i), mse[:i])
    return line, loss_line


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


def write_to_json(theta0: float, theta1: float) -> None:
    with open("model.json", "w") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f)


def min_max_normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize 2D tensor with feature scaling method

    :param data: the 2D tensor to be normalized
    :type data: np.ndarray
    :return: normalized tensor
    :rtype: np.ndarray
    """
    minimum1 = min(data[0])
    diff1 = max(data[0]) - minimum1
    minimum2 = min(data[1])
    diff2 = max(data[1]) - minimum2
    res1 = [((x - minimum1) / diff1) for x in data[0]]
    res2 = [((y - minimum2) / diff2) for y in data[1]]
    return np.asarray([res1, res2], dtype=float)


def scale_coefficients(
        theta0: float,
        theta1: float) -> tuple[float, float]:
    th1: float = theta1 * (max_y - min_y) / (max_x - min_x)
    th0: float = min_y + theta0 * (max_y - min_y) - th1 * min_x
    return th0, th1


def linear_regression_train(
        data: np.ndarray[np.ndarray[float]],
        learning_rate: float = 0.02,
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
    div = -2 / n
    for i in range(epochs):
        y_pred = m * x + b
        diff_vectors = y - y_pred
        dm = div * np.sum(x * diff_vectors)
        db = div * np.sum(diff_vectors)

        m -= learning_rate * dm
        b -= learning_rate * db

        if i % 10 == 0:
            mse.append((1 / n) * np.sum((y - y_pred) ** 2))
            theta.append((m, b))
    th0, th1 = scale_coefficients(b, m)
    write_to_json(th0, th1)


def linear_regression_train_subject(
        data: np.ndarray[np.ndarray[float]],
        learning_rate: float = 0.05,
        epochs: int = 1000) -> None:
    """
    Trains the model with the data provided to this function. This version of
    the function uses the formulas presented in the subject instead of the ones
    calculated from the partial derivatives of the mean square error formula.

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
    theta0, theta1 = .0, .0
    n = len(x)
    div = 1 / n
    for i in range(epochs):
        y_pred = theta0 + theta1 * x
        diff_vectors = y_pred - y
        theta0 -= learning_rate * div * np.sum(diff_vectors)
        theta1 -= learning_rate * div * np.sum(diff_vectors * x)

        if i % 10 == 0:
            mse.append((1 / n) * np.sum((y - y_pred) ** 2))
            theta.append((theta1, theta0))
    th0, th1 = scale_coefficients(theta0, theta1)
    write_to_json(th0, th1)


def main():
    """
    The program accepts 1 argument which defines
    the path to the training data
    """
    """ if (len(sys.argv) != 2
            or not os.path.isfile(sys.argv[1])
            or not sys.argv[1].endswith(".csv")):
        log_error("Provide a valid path to a .csv file")
        exit(Errors.INVALID_ARGUMENT.value) """
    (data, headers) = read_csv("./data/data.csv")
    global min_x
    global max_x
    global min_y
    global max_y
    min_x = np.min(data[0])
    max_x = np.max(data[0])
    min_y = np.min(data[1])
    max_y = np.max(data[1])
    norm_data = min_max_normalize(data)
    global fig_data
    global min_norm_data
    global max_norm_data
    fig_data = norm_data
    min_norm_data = min(fig_data[0])
    max_norm_data = max(fig_data[0])
    linear_regression_train_subject(norm_data)
    ax1.set_box_aspect(1)
    ax1.set_xlim([-0.1, 1.1])
    ax1.set_ylim([-0.1, 1.1])
    ax1.set_xlabel(headers[0])
    ax1.set_ylabel(headers[1])
    ax2.set_box_aspect(1)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, max(mse) * 1.1)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Squared Error (MSE)")
    plt.subplots_adjust(wspace=0.5, hspace=0.2)
    ax1.scatter(fig_data[0], fig_data[1])
    anim = animation.FuncAnimation(fig=fig, func=animate, frames=len(mse))
    anim.save("gd.gif", fps=30)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# %%
