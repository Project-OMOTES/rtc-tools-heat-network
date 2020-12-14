from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem


class TestBufferHistory(TestCase):
    def test_buffer_history(self):
        import models.simple_buffer.src.simple_buffer as simple_buffer
        from models.simple_buffer.src.simple_buffer import (
            HeatBufferNoHistory,
            HeatBufferHistory,
            HeatBufferHistoryStoredHeat,
        )

        base_folder = Path(simple_buffer.__file__).resolve().parent.parent

        nohistory = run_optimization_problem(HeatBufferNoHistory, base_folder=base_folder)
        history = run_optimization_problem(HeatBufferHistory, base_folder=base_folder)
        historystoredheat = run_optimization_problem(
            HeatBufferHistoryStoredHeat, base_folder=base_folder
        )

        nohistory_results = nohistory.extract_results()
        history_results = history.extract_results()
        historystoredheat_results = historystoredheat.extract_results()

        bufferheat_nohistory = nohistory_results["buffer.Heat_buffer"]
        bufferheat_history = history_results["buffer.Heat_buffer"]
        bufferheat_historystoredheat = historystoredheat_results["buffer.Heat_buffer"]

        np.testing.assert_array_equal(0.0, bufferheat_nohistory[0])
        np.testing.assert_array_less(bufferheat_history[0], 0.0)
        np.testing.assert_array_equal(bufferheat_history, bufferheat_historystoredheat)
