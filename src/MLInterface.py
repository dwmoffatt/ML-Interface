"""
Machine Learning Command Line Interface
"""
import copy
import logging
import math

import matplotlib.pyplot as plt
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

    def gradient_descent(self, algo: REGRESSION.RegressionTypes, w_in: np.ndarray, b_in: float, alpha: float, num_iters: int) -> (np.ndarray, int):
        """
        Performs gradient descent to fit w,b. Updates w,b by taking num_iters gradient steps with learning rate alpha

        Save history of cost and hist of params

        :param:
            algo (RegressionTypes) : What type of algorithm to use with gradient descent
            w_in (ndarray (n,))    : Initial model parameters
            b_in (float)          : Initial model parameter
            alpha (float)          : Learning rate
            num_iters (int)        : Number of iterations to run gradient descent

        :returns:
            w (ndarray (n,)) : Updated value of parameter after running gradient descent
            b (int)       : Updated value of parameter after running gradient descent
        """

        # An array to store cost J and w's at each iteration primarily for graphing later
        j_history = []
        p_history = []
        w = copy.deepcopy(w_in)
        b = b_in

        match algo:
            case REGRESSION.RegressionTypes.LINEAR:
                ml_model = REGRESSION.LinearRegression(x_train=self.__x_train, y_train=self.__y_train, weights=w, bias=b)
            case _:
                raise ValueError("Algo param not set to valid type")

        for i in range(num_iters):
            # Calculate the gradient and update the parameters using gradient_function
            dj_dw, dj_db = ml_model.compute_gradient()

            # Update Parameters using equation (3) above
            ml_model.bias = ml_model.bias - alpha * dj_db
            ml_model.weights = ml_model.weights - alpha * dj_dw

            # Save cost J at each iteration
            if i < 1000000:  # prevent resource exhaustion
                j_history.append(ml_model.compute_cost())
                p_history.append([ml_model.weights, ml_model.bias])
            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % math.ceil(num_iters / 10) == 0:
                print(
                    f"Iteration {i:4d}: Cost {j_history[-1]:8.2f} ",
                    f"dj_dw: {dj_dw}, dj_db: {dj_db}  ",
                    f"w: {ml_model.weights}, b:{ml_model.bias}",
                )

        self.__hist_of_cost = j_history
        self.__hist_of_params = p_history

        return ml_model.weights, ml_model.bias

    def plot_cost_vs_iteration(self) -> None:
        """
        Plot cost vs iteration

        :returns:
            Nothing
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
        ax1.plot(self.__hist_of_cost)
        ax2.plot(100 + np.arange(len(self.__hist_of_cost[100:])), self.__hist_of_cost[100:])
        ax1.set_title("Cost vs. iteration")
        ax2.set_title("Cost vs. iteration (tail)")
        ax1.set_ylabel("Cost")
        ax2.set_ylabel("Cost")
        ax1.set_xlabel("Iteration Step")
        ax2.set_xlabel("Iteration Step")
        plt.show()


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
