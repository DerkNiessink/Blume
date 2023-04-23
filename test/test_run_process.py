import unittest
from datetime import datetime
import os
import shutil
import sys

from blume.run import sweep_T, save, new_folder
from blume.process import read, compute


class TestRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.now = datetime.now().strftime("%d-%m %H:%M")
        cls.chi_list = [2, 4]
        cls.L_list = [4, 8]
        dir = new_folder()

        for chi in cls.chi_list:
            data = sweep_T(chi=chi, T_range=[2.5, 2.6, 2.6], bar=False)
            save(data, dir, msg=False)

        for L in cls.L_list:
            data = sweep_T(
                T_range=(2.5, 2.6), step=0.001, max_steps=L, b_c=True, bar=False
            )
            save(data, dir, msg=False)

    def test_new_folder(self):
        """
        Test that there is a new dir in `data` with the current datetime as
        name.
        """
        self.assertTrue(os.path.isdir(f"data/{TestRun.now}"))

    def test_save(self):
        """
        Test that the data is saved in the new directory with the right name.
        """
        for L in TestRun.L_list:
            with self.subTest():
                self.assertTrue(os.path.isfile(f"data/{TestRun.now}/L{L}.json"))

        for chi in TestRun.chi_list:
            with self.subTest():
                self.assertTrue(os.path.isfile(f"data/{TestRun.now}/chi{chi}.json"))

    def test_contents(self):
        """
        Read the data and check that it contains a dictionary with the right
        keys and non empty values.
        """
        for chi in TestRun.chi_list:
            data = read(TestRun.now, chi)
            self.assertTrue(len(data) > 2)

            for key in data:
                if isinstance(data[key], list):
                    # Check that the lists are not empty
                    with self.subTest():
                        self.assertFalse(data[key] == [])

    @classmethod
    def tearDownClass(cls):
        """
        Remove the new directory after all tests.
        """
        shutil.rmtree(f"data/{cls.now}")
