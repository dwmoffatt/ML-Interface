"""
Machine Learning Command Line Interface
"""
import logging
import math
import numpy as np

# import matplotlib.pyplot as plt

import learning as ml_learning

REGRESSION = ml_learning.supervised.regression


class MLInterface:
    __x_train: np.ndarray = np.zeros(2)
    __y_train: np.ndarray = np.zeros(2)
    __hist_of_cost: list = list()
    __hist_of_params: list = list()

    def __init__(self):
        """
        ML Interface Initialization
        """
        ...

    @property
    def x_train(self) -> np.ndarray:
        return self.__x_train

    @x_train.setter
    def x_train(self, x_train: np.ndarray) -> None:
        if isinstance(x_train, np.ndarray) is not True:
            raise TypeError("x_train needs to be of type np.ndarray")
        self.__x_train = x_train

    @property
    def y_train(self) -> np.ndarray:
        return self.__y_train

    @y_train.setter
    def y_train(self, y_train: np.ndarray) -> None:
        if isinstance(y_train, np.ndarray) is not True:
            raise TypeError("y_train needs to be of type np.ndarray")
        self.__y_train = y_train

    def gradient_descent(self, algo: REGRESSION.RegressionTypes, w_in, b_in, alpha: float, num_iters: int):
        """
        Performs gradient descent to fit w,b. Updates w,b by taking
        num_iters gradient steps with learning rate alpha

        Args:
          w_in,b_in (scalar): initial values of model parameters
          alpha (float):     Learning rate
          num_iters (int):   number of iterations to run gradient descent

        Returns:
          w (scalar): Updated value of parameter after running gradient descent
          b (scalar): Updated value of parameter after running gradient descent
          j_history (List): History of cost values
          p_history (list): History of parameters [w,b]
        """

        # An array to store cost J and w's at each iteration primarily for graphing later
        j_history = []
        p_history = []

        match algo:
            case REGRESSION.RegressionTypes.LINEAR:
                ml_model = REGRESSION.LinearRegression(x_train=self.__x_train, y_train=self.__y_train, weights=w_in, bias=b_in)
            case _:
                raise ValueError("Algo param not set to valid type")

        for i in range(num_iters):
            # Calculate the gradient and update the parameters using gradient_function
            dj_dw, dj_db = ml_model.compute_gradient()

            # Update Parameters using equation (3) above
            ml_model.bias = ml_model.bias - alpha * dj_db
            ml_model.weights = ml_model.weights - alpha * dj_dw

            # Save cost J at each iteration
            if i < 100000:  # prevent resource exhaustion
                j_history.append(ml_model.compute_cost())
                p_history.append([ml_model.weights, ml_model.bias])
            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % math.ceil(num_iters / 10) == 0:
                print(
                    f"Iteration {i:4}: Cost {j_history[-1]:0.2e} ",
                    f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                    f"w: {ml_model.weights: 0.3e}, b:{ml_model.bias: 0.5e}",
                )

        self.__hist_of_cost = j_history
        self.__hist_of_params = p_history

        return ml_model.weights, ml_model.bias


if __name__ in ("__main__", "__builtin__"):
    logging.basicConfig(
        filename="Interface.log",
        format="[%(asctime)s][%(levelname)s] - %(message)s",
        filemode="w",
        level=logging.INFO,
    )

    print(
        "# ML Learning Interface\n",
        "# ---------------------\n",
        f"# Version: {ml_learning.__version__}",
    )

    ml = MLInterface()
