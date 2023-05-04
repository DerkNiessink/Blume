import unittest
from datetime import datetime
import os
import shutil

from blume.run import ModelParameters, Results
from blume.process import read, compute


class TestRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.now = datetime.now().strftime("%d-%m %H:%M")
        cls.chi_list = [2, 4]
        cls.max_steps_list = [4, 8]

        chi_test = Results("chi", TestRun.chi_list)
        max_steps_test = Results("max_steps", TestRun.max_steps_list)
        fixed_test = Results()

        params = ModelParameters(T_range=[2.5, 2.6, 2.6], bar=False)
        chi_test.get(params)
        max_steps_test.get(params)
        fixed_test.get(
            ModelParameters(T_range=[2.5, 2.7], b_c=True, fixed=True, bar=False)
        )

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
        for max_step in TestRun.max_steps_list:
            with self.subTest():
                self.assertTrue(
                    os.path.isfile(f"data/{TestRun.now}/max_steps{max_step}.json")
                )

        for chi in TestRun.chi_list:
            with self.subTest():
                self.assertTrue(os.path.isfile(f"data/{TestRun.now}/chi{chi}.json"))

        with self.subTest():
            self.assertTrue(os.path.isfile(f"data/{TestRun.now}/data.json"))

    def test_contents(self):
        """
        Read the data and check that it contains a dictionary with the right
        keys and non empty values.
        """
        for chi in TestRun.chi_list:
            data = read(TestRun.now, f"chi{chi}")
            with self.subTest():
                self.assertTrue(len(data) > 2)

            # Check that it does not save the fixed edges.
            with self.subTest():
                self.assertTrue(data["converged fixed edges"][1] == None)

            for key in data:
                if isinstance(data[key], list):
                    # Check that the lists are not empty
                    with self.subTest():
                        self.assertFalse(data[key] == [])

        data = read(TestRun.now, "data")
        # Check that it saves the fixed edges.
        with self.subTest():
            self.assertFalse(data["converged fixed edges"][1] == None)

    @classmethod
    def tearDownClass(cls):
        """
        Remove the new directory after all tests.
        """
        shutil.rmtree(f"data/{cls.now}")
