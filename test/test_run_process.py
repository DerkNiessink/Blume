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
        cls._now = datetime.now().strftime("%d-%m %H:%M")

    def setUp(self):
        self.now = TestRun._now
        self.chi_list = [2, 4]
        self.L_list = [4, 8]
        dir = new_folder()

        for chi in self.chi_list:
            data = sweep_T(chi=chi, T_range=(2.5, 2.6), step=0.01)
            save(data, dir)

        for L in self.L_list:
            data = sweep_T(T_range=(2.5, 2.6), step=0.01, max_steps=L, b_c=True)
            save(data, dir)

    def test_new_folder(self):
        """
        Test that there is a new dir in `data` with the current datetime as
        name.
        """
        self.assertTrue(os.path.isdir(f"data/{self.now}"))

    def test_save(self):
        """
        Test that the data is saved in the new directory with the right name.
        """
        for L in self.L_list:
            with self.subTest():
                self.assertTrue(os.path.isfile(f"data/{self.now}/L{L}.json"))

        for chi in self.chi_list:
            with self.subTest():
                self.assertTrue(os.path.isfile(f"data/{self.now}/chi{chi}.json"))

    def test_contents(self):
        """
        Read the data and check that it contains a dictionary with the right
        keys and non empty values.
        """
        for chi in self.chi_list:
            data = read(self.now, chi)
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
        shutil.rmtree(f"data/{cls._now}")
