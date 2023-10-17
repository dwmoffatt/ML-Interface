"""
Machine Learning Command Line Interface
"""
import logging
import math
import numpy as np


class MLInterface:
    __x_train = np.zeros(2)
    __y_train = np.zeros(2)

    def __init__(self):
        """ """
        ...

    @property
    def x_train(self) -> np.ndarray:
        return self.__x_train

    @x_train.setter
    def x_train(self, x_train: np.ndarray) -> None:
        self.__x_train = x_train

    @property
    def y_train(self) -> np.ndarray:
        return self.__y_train

    @y_train.setter
    def y_train(self, y_train: np.ndarray) -> None:
        self.__y_train = y_train

    @staticmethod
    def __compute_cost(x: np.ndarray, y: np.ndarray, w, b) -> float:
        """
        Computes the cost function for linear regression.

        Args:
          x (ndarray (m,)): Data, m examples
          y (ndarray (m,)): target values
          w,b (scalar)    : model parameters

        Returns
            total_cost (float): The cost of using w,b as the parameters for linear regression
                   to fit the data points in x and y
        """
        m = x.shape[0]
        cost = 0

        for i in range(m):
            f_wb = w * x[i] + b
            cost = cost + (f_wb - y[i]) ** 2
        total_cost = 1 / (2 * m) * cost

        return total_cost

    @staticmethod
    def __compute_gradient(x: np.ndarray, y: np.ndarray, w, b) -> tuple:
        """
        Computes the gradient for linear regression

        Args:
          x (ndarray (m,)): Data, m examples
          y (ndarray (m,)): target values
          w,b (scalar)    : model parameters
        Returns
          dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
          dj_db (scalar): The gradient of the cost w.r.t. the parameter b
        """

        # Number of training examples
        m = x.shape[0]
        dj_dw = 0
        dj_db = 0

        for i in range(m):
            f_wb = w * x[i] + b
            dj_dw_i = (f_wb - y[i]) * x[i]
            dj_db_i = f_wb - y[i]
            dj_db += dj_db_i
            dj_dw += dj_dw_i
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db

    def gradient_descent(self, w_in, b_in, alpha: float, num_iters: int):
        """
        Performs gradient descent to fit w,b. Updates w,b by taking
        num_iters gradient steps with learning rate alpha

        Args:
          x (ndarray (m,))  : Data, m examples
          y (ndarray (m,))  : target values
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
        b = b_in
        w = w_in

        for i in range(num_iters):
            # Calculate the gradient and update the parameters using gradient_function
            dj_dw, dj_db = self.__compute_gradient(self.__x_train, self.__y_train, w, b)

            # Update Parameters using equation (3) above
            b = b - alpha * dj_db
            w = w - alpha * dj_dw

            # Save cost J at each iteration
            if i < 100000:  # prevent resource exhaustion
                j_history.append(self.__compute_cost(self.__x_train, self.__y_train, w, b))
                p_history.append([w, b])
            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % math.ceil(num_iters / 10) == 0:
                print(f"Iteration {i:4}: Cost {j_history[-1]:0.2e} ", f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ", f"w: {w: 0.3e}, b:{b: 0.5e}")

        return w, b, j_history, p_history  # return w and J,w history for graphing


if __name__ in ("__main__", "__builtin__"):
    logging.basicConfig(
        filename="Interface.log",
        format="[%(asctime)s][%(levelname)s] - %(message)s",
        filemode="w",
        level=logging.INFO,
    )

    ml = MLInterface()
