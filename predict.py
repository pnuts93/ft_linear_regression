import json
import sys


def estimatePrice(mileage: float, theta0: float, theta1: float) -> float:
    """
    Attempts to predict the price of a car knowing its mileage

    :param mileage: the mileage of the car
    :type mileage: float
    :param theta0: the x-intercept of the regression line
    :type theta0: float
    :param theta1: the slope of the regression line
    :type theta1: float
    :returns: the estimated price
    :rtype: float
    """
    return theta0 + theta1 * mileage


def load_model() -> tuple[float, float]:
    """
    Reads `model.json` and extracts theta0 and theta1

    :returns: theta0 and theta1
    :rtype: tuple[float, float]
    """
    theta0: float
    theta1: float
    with open("model.json") as f:
        file = json.load(f)
        theta0 = float(file.get("theta0"))
        theta1 = float(file.get("theta1"))
    return theta0, theta1


def main():
    """
    Attempts to predict the price of a car knowing its mileage
    """
    try:
        theta0, theta1 = load_model()
        print(estimatePrice(float(sys.argv[1]), theta0, theta1))
    except Exception:
        print("Error")


if __name__ == "__main__":
    main()
