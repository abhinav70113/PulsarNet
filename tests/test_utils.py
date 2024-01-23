import unittest
import numpy as np
from utils import overlapping_windows, compute_features, filter_predictions, parse_inf_file, load_config, parse_pulsarnet_output_file
import tempfile
import os

class TestUtils(unittest.TestCase):

    # Tests for overlapping_windows
    def test_overlapping_windows_valid(self):
        input_array = np.array([1, 2, 3, 4, 5])
        step_size = 1
        chunk_size = 3
        expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        result = overlapping_windows(input_array, step_size, chunk_size)
        np.testing.assert_array_equal(result, expected)

    def test_overlapping_windows_edge(self):
        input_array = np.array([1, 2, 3])
        step_size = 2
        chunk_size = 4
        expected = np.array([])
        result = overlapping_windows(input_array, step_size, chunk_size)
        self.assertEqual(len(result), 0)

    # Tests for compute_features
    def test_compute_features_valid(self):
        arr = np.array([1, 2, 3, 4])
        expected = [2.5, 10.0, 1.25, 2.23606797749979]  # mean, sum, variance, sqrt of mean square
        result = compute_features(arr)
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e)

    def test_compute_features_empty(self):
        arr = np.array([])
        expected = [np.nan, 0, np.nan, np.nan]  # mean, sum, variance, sqrt of mean square
        result = compute_features(arr)
        for r, e in zip(result, expected):
            self.assertTrue(np.isnan(r) if np.isnan(e) else r == e)

    # Tests for filter_predictions
    # Note: This requires more specific test cases based on your algorithm's logic.

    # Tests for parse_inf_file
    def test_parse_inf_file(self):
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp:
            tmp.write("key1=value1\nkey2=value2 # comment\n")
            tmp_path = tmp.name
        
        expected = {"key1": "value1", "key2": "value2"}
        result = parse_inf_file(tmp_path)
        os.remove(tmp_path)

        self.assertEqual(result, expected)

    # Tests for load_config
    # Note: Similar to parse_inf_file, requires a temporary file with config content.

    # Tests for parse_pulsarnet_output_file
    # Note: This function depends heavily on the specific format of your log files.

if __name__ == '__main__':
    unittest.main()
