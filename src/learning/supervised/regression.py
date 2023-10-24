"""
Regression
"""
import numpy as np
from enum import Enum


class RegressionTypes(Enum):
    LINEAR = 1


class LinearRegression:
    __x_train: np.ndarray = np.zeros(2)
    __y_train: np.ndarray = np.zeros(2)
    __weights: np.ndarray = np.zeros(2)
    __bias: int = 0

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray, bias: int = 0) -> None:
        self.__x_train = x_train
        self.__y_train = y_train
        self.__weights = weights
        self.__bias = bias

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

    @property
    def weights(self) -> np.ndarray:
        return self.__weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        self.__weights = weights

    @property
    def bias(self) -> int:
        return self.__bias

    @bias.setter
    def bias(self, bias: int) -> None:
        self.__bias = bias

    def compute_cost(self) -> float:
        """
        Computes the cost function for linear regression based on:
            x_train (ndarray (m,n)) : Data values
            y_train (ndarray (m,))  : Target values
            weights (ndarray (n,))  : Model parameters
            bias    (int)           : Model parameters

        :return:
            total_cost (float): The cost of using w,b as the parameters for linear regression
                to fit the data points in x and y
        """
        m = self.__x_train.shape[0]
        cost = 0

        for i in range(m):
            f_wb_i = np.dot(self.__x_train[i], self.__weights) + self.__bias
            cost = cost + (f_wb_i - self.__y_train[i]) ** 2
        total_cost = cost / (2 * m)

        return total_cost

    def compute_gradient(self) -> (np.ndarray, int):
        """
        Computes the gradient for linear regression
            x_train (ndarray (m,n)) : Data values
            y_train (ndarray (m,))  : Target values
            weights (ndarray (n,))  : Model parameters
            bias    (int)           : Model parameters

        :return:
            dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w
            dj_db (int):          The gradient of the cost w.r.t. the parameter b
        """

        # Number of training examples
        m, n = self.__x_train.shape
        dj_dw = np.zeros((n,))
        dj_db = 0

        for i in range(m):
            err = (np.dot(self.__x_train[i], self.__weights) + self.__bias) - self.__y_train[i]
            for j in range(n):
                dj_dw[j] = dj_dw[j] + err * self.__x_train[i, j]
            dj_db = dj_db + err
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db
