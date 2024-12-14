import json
import sys


def estimatePrice(mileage: float, theta0: float, theta1: float) -> float:
    return theta0 + theta1 * mileage


def load_model() -> tuple[float, float]:
    theta0: float
    theta1: float
    with open("model.json") as f:
        file = json.load(f)
        theta0 = float(file.get("theta0"))
        theta1 = float(file.get("theta1"))
    return theta0, theta1


def main():
    try:
        theta0, theta1 = load_model()
        print(estimatePrice(float(sys.argv[1]), theta0, theta1))
    except Exception:
        print("Error")


if __name__ == "__main__":
    main()
