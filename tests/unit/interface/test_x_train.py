"""
Tests x_train property of MLInterface Class
"""
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

    def test_x_train_setter(self):
        """
        Tests that x_train can be written to, and only accepts values of np.ndarray type

        :return:
        """
        valid_data = np.array([400.0, 300.00, 500.00])

        self.ml_interface.x_train = valid_data
        check.is_(self.ml_interface.x_train, valid_data)

        invalid_data = 1
        with check.raises(TypeError):
            self.ml_interface.x_train = invalid_data

    def teardown_method(self, method):
        del self.ml_interface

    # @classmethod
    # def teardown_class(cls):
