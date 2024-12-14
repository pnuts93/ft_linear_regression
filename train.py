# %%
import sys
import json
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
min_plot_data: float
max_plot_data: float


class bcolors:
    """
    Provides a list of colors that can be used in STDOUT
    """
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
    """
    Provides the logic to animate the graphs

    :param i: iteration number
    :type i: int
    :returns: animated elements to be plotted
    """
    line.set_data(
        [min_plot_data,
         max_plot_data],
        [theta[i][0] * min_plot_data + theta[i][1],
         theta[i][0] * max_plot_data + theta[i][1]])
    loss_line.set_data(range(0, i * 10, 10), mse[:i])
    return line, loss_line


def plot_data(data: np.ndarray, headers: list[str], epochs: int):
    """
    Creates and saves the animated plots

    :param data: 2D tensor used for the training of the model
    :type data: np.ndarray
    :param headers: the headers of the dataset
    :type headers: list[str]
    """
    global min_plot_data
    global max_plot_data
    min_plot_data = np.min(data)
    max_plot_data = np.max(data)
    ax1.set_box_aspect(1)
    ax1.set_xlim([-0.1, 1.1])
    ax1.set_ylim([-0.1, 1.1])
    ax1.set_xlabel(headers[0])
    ax1.set_ylabel(headers[1])
    ax2.set_box_aspect(1)
    ax2.set_xlim(0, epochs)
    ax2.set_ylim(0, max(mse) * 1.1)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Squared Error (MSE)")
    plt.subplots_adjust(wspace=0.5, hspace=0.2)
    ax1.scatter(data[0], data[1])
    anim = animation.FuncAnimation(fig=fig, func=animate, frames=len(mse))
    anim.save("gd.gif", fps=30)
    plt.tight_layout()
    plt.show()


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
    """
    Writes the two coefficients to a json file named "model.json"

    :param theta0: the x intercept of the regression line
    :type theta0: float
    :param theta1: the slope of the regression line
    :type theta1: float
    """
    with open("model.json", "w") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f)


def min_max_scale(data: np.ndarray) -> np.ndarray:
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
        theta1: float,
        data: np.ndarray) -> tuple[float, float]:
    """
    Scales the coefficient so that they can be used to calculate the predicted
    dependent variable. This should be used only if the coefficients were
    calculated on a dataset where min-max scaling was used

    :param theta0: the x intercept of the regression line
    :type theta0: float
    :param theta1: the slope of the regression line
    :type theta1: float
    :param data: the original data (before scaling)
    :type data: np.ndarray
    :return: the scaled coefficients
    :rtype: tuple[float, float]
    """
    min_x: float = np.min(data[0])
    max_x: float = np.max(data[0])
    min_y: float = np.min(data[1])
    max_y: float = np.max(data[1])
    th1: float = theta1 * (max_y - min_y) / (max_x - min_x)
    th0: float = min_y + theta0 * (max_y - min_y) - th1 * min_x
    return th0, th1


def linear_regression_train(
        data: np.ndarray[np.ndarray[float]],
        headers: list[str],
        learning_rate: float = 0.02,
        epochs: int = 1000,
        should_scale_data: bool = True,
        should_plot_data: bool = True) -> None:
    """
    Trains the model with the data provided to this function

    :param data: 2D tensor. The data is expected to be divided in two
                sub-arrays
    :type data: np.ndarray
    :param headers: the headers of the training dataset. Must be provided if
                the data is going to be plotted
    :type headers: list[str]
    :param alpha: the learning rate of the model, defaults to 0.01
    :type alpha: float
    :param epochs: the epochs (number of iterations) used by the model to
                get fine-tuned, defaults to 1000
    :type epochs: int
    :param should_scale_data: defines if min max scaling will be used on the
                dataset, defaults to True
    :type should_scale_data: bool
    :param should_plot_data: defines if the data will be plotted,
                defaults to True
    :type should_plot_data: bool
    """
    if len(data) != 2:
        raise Exception("The function expects a 2D tensor")
    if should_plot_data and len(headers) != 2:
        raise Exception("Headers should be provided in order to create plot")
    if should_scale_data:
        original_data = data
        data = min_max_scale(data)
    (x, y) = data
    m, b = .0, .0
    n = len(x)
    for i in range(epochs):
        y_pred = m * x + b
        dm = (-2 / n) * np.sum(x * (y - y_pred))
        db = (-2 / n) * np.sum((y - y_pred))

        m -= learning_rate * dm
        b -= learning_rate * db

        if should_plot_data and i % 10 == 0:
            mse.append((1 / n) * np.sum((y - y_pred) ** 2))
            theta.append((m, b))
    if should_scale_data:
        b, m = scale_coefficients(b, m, original_data)
    write_to_json(b, m)
    if should_plot_data:
        plot_data(data, headers, epochs)


def linear_regression_train_subject(
        data: np.ndarray[np.ndarray[float]],
        headers: list[str],
        learning_rate: float = 0.05,
        epochs: int = 1000,
        should_scale_data: bool = True,
        should_plot_data: bool = True) -> None:
    """
    Trains the model with the data provided to this function. This version of
    the function uses the formulas presented in the subject instead of the ones
    calculated from the partial derivatives of the mean square error formula.

    :param data: 2D tensor. The data is expected to be divided in two
                sub-arrays
    :type data: np.ndarray
    :param headers: the headers of the training dataset. Must be provided if
                the data is going to be plotted
    :type headers: list[str]
    :param learning_rate: the learning rate of the model, defaults to 0.01
    :type learning_rate: float
    :param epochs: the epochs (number of iterations) used by the model to
                get fine-tuned, defaults to 1000
    :type epochs: int
    :param should_scale_data: defines if min max scaling will be used on the
                dataset, defaults to True
    :type should_scale_data: bool
    :param should_plot_data: defines if the data will be plotted,
                defaults to True
    :type should_plot_data: bool
    """
    if len(data) != 2:
        raise IndexError("The function expects a 2D tensor")
    if should_plot_data and len(headers) != 2:
        raise IndexError("Headers should be provided in order to create plot")
    if should_scale_data:
        original_data = data
        data = min_max_scale(data)
    (x, y) = data
    theta0, theta1 = .0, .0
    n = len(x)
    for i in range(epochs):
        y_pred = theta0 + theta1 * x
        theta0 -= learning_rate * (1 / n) * np.sum((y_pred - y))
        theta1 -= learning_rate * (1 / n) * np.sum((y_pred - y) * x)

        if should_plot_data and i % 10 == 0:
            mse.append((1 / n) * np.sum((y - y_pred) ** 2))
            theta.append((theta1, theta0))
    if should_scale_data:
        theta0, theta1 = scale_coefficients(theta0, theta1, original_data)
    write_to_json(theta0, theta1)
    if should_plot_data:
        plot_data(data, headers, epochs)


def main():
    """
    The program calculates the coefficients used in linear regression with
    gradient descent and exports them in `model.json`
    """
    try:
        (data, headers) = read_csv("./data/data.csv")
        linear_regression_train_subject(data, headers)
    except FileNotFoundError:
        log_error("data file does not exist")
    except IndexError as index_error:
        log_error(index_error.args[0])


if __name__ == "__main__":
    main()

# %%
