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
    __weights: int = 0
    __bias: int = 0

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, weights: int = 0, bias: int = 0) -> None:
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
    def bias(self) -> int:
        return self.__bias

    @bias.setter
    def bias(self, bias: int) -> None:
        self.__bias = bias

    @property
    def weights(self) -> int:
        return self.__weights

    @weights.setter
    def weights(self, weights: int) -> None:
        self.__weights = weights

    def compute_cost(self) -> float:
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
        m = self.__x_train.shape[0]
        cost = 0

        for i in range(m):
            f_wb = self.__weights * self.__x_train[i] + self.__bias
            cost = cost + (f_wb - self.__y_train[i]) ** 2
        total_cost = 1 / (2 * m) * cost

        return total_cost

    def compute_gradient(self) -> tuple:
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
        m = self.__x_train.shape[0]
        dj_dw = 0
        dj_db = 0

        for i in range(m):
            f_wb = self.__weights * self.__x_train[i] + self.__bias
            dj_dw_i = (f_wb - self.__y_train[i]) * self.__x_train[i]
            dj_db_i = f_wb - self.__y_train[i]
            dj_db += dj_db_i
            dj_dw += dj_dw_i
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db
