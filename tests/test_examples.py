import inspect
import os
import subprocess
import sys
from unittest import TestCase


class ExamplesCollection:
    def __init__(self):
        self.errors_detected = {}
        for example_folder in self.examples_folders:
            # initialize with failures
            example_path = os.path.join(self.examples_path, example_folder, "src", "example.py")
            self.errors_detected[example_path] = True

    def local_function(self):
        pass

    @property
    def examples_path(self):
        return os.path.join(
            os.path.dirname(os.path.abspath(inspect.getsourcefile(self.local_function))),
            "..",
            "examples",
        )

    @property
    def examples_folders(self):
        folders = []
        for dirpath, dirnames, _ in os.walk(self.examples_path):
            if "src" in dirnames:
                folders.append(os.path.relpath(dirpath, self.examples_path))
        return folders


class TestExamples(TestCase):
    def run_examples(self, ec):
        env = sys.executable
        for example_path in ec.errors_detected.keys():
            try:
                subprocess.check_output([env, example_path])
            except Exception:
                ec.errors_detected[example_path] = True
            else:
                ec.errors_detected[example_path] = False

    def test_examples(self):
        ec = ExamplesCollection()

        self.run_examples(ec)

        for example_path, error_detected in ec.errors_detected.items():
            if error_detected:
                print("An error occured while running example '{}'.".format(example_path))
            else:
                print("No errors occured while running example '{}'.".format(example_path))

        self.assertFalse(any(ec.errors_detected.values()))
