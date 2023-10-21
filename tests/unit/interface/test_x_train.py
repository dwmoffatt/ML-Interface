import numpy as np
from pytest_check import check
from src.MLInterface import MLInterface


class TestXTrain:
    # @classmethod
    # def setup_class(cls):

    def setup_method(self, method):
        self.ml_interface = MLInterface()

    def test_x_train_default(self):
        """
        Tests that x_train has a default value of type np.ndarray

        :return:
        """
        check.is_instance(self.ml_interface.x_train, np.ndarray)

    def teardown_method(self, method):
        del self.ml_interface

    # @classmethod
    # def teardown_class(cls):
