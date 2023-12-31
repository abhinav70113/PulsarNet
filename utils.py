"""
utils.py

This module offers utility functions that can be reused throughout the project.
It provides functionalities to extract overlapping windows from an array, compute features, and filter predictions.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
import json
import re

def overlapping_windows(input_array: np.ndarray, step_size: int, chunk_size: int) -> np.ndarray:
    """
    Return overlapping windows from the provided array.

    Parameters:
    - input_array (np.ndarray): Input array from which overlapping windows are to be extracted.
    - step_size (int): The step size or stride between each window.
    - chunk_size (int): The size of each window or chunk.

    Returns:
    - np.ndarray: Array containing overlapping windows.
    """
    # Calculate the number of windows
    num_windows = 1 + (len(input_array) - chunk_size) // step_size
    
    # Create the overlapping windows using as_strided
    output = as_strided(
        input_array,
        shape=(num_windows, chunk_size),
        strides=(input_array.strides[0] * step_size, input_array.strides[0])
    )
    
    return output

def compute_features(arr: np.ndarray) -> list:
    """
    Compute a feature set for the provided array.

    Parameters:
    - arr (np.ndarray): Input array from which features are to be computed.

    Returns:
    - list: Feature set corresponding to the input array.
    """
    return [np.mean(arr), np.sum(arr), np.var(arr), np.sqrt(np.mean(np.square(arr)))]

def filter_predictions(decision_values: np.ndarray, step_size: int, freq_ind_start: int, roll_number: int) -> np.ndarray:
    """
    Filter and return predictions based on the provided decision values.
    This algo will have a problem if one 50 indices wide chunk contains two signals.

    Parameters:
    - decision_values (np.ndarray): Decision values based on which predictions are to be filtered.
    - step_size (int): Step size or stride between predictions.
    - freq_ind_start (int): Start index for frequency-based filtering.
    - roll_number (int): The roll number for the rolling window.

    Returns:
    - np.ndarray: Array containing filtered predictions.
    """
    sorted_indices = np.argsort(decision_values)[::-1]
    sure_signals = []
    marked_areas = set()

    for idx in sorted_indices:
        # Check for decision function value greater than 0
        if (decision_values[idx] <= 0) or (idx in sure_signals):
            continue
        
        sure_signal_idx_to_append = idx

        # Default marking position
        original_position = (idx + 1) * step_size + freq_ind_start - roll_number

        prev_decision = decision_values[idx - 1] if idx - 1 >= 0 else None
        next_decision = decision_values[idx + 1] if idx + 1 < len(decision_values) else None

        # Additional conditions for marking
        if next_decision and next_decision > 0 and (not prev_decision or prev_decision < 0):
            original_position = (idx + 1) * step_size + freq_ind_start
            sure_signal_idx_to_append = idx
        elif prev_decision and prev_decision > 0 and (not next_decision or next_decision < 0):
            original_position = idx * step_size + freq_ind_start
            sure_signal_idx_to_append = idx - 1
        elif prev_decision and next_decision and prev_decision > 0 and next_decision > 0:
            if abs(decision_values[idx] - prev_decision) < abs(decision_values[idx] - next_decision):
                original_position = idx * step_size + freq_ind_start
                sure_signal_idx_to_append = idx - 1
            else:
                original_position = (idx + 1) * step_size + freq_ind_start
                sure_signal_idx_to_append = idx

        elif (not prev_decision or prev_decision < 0) and (not next_decision or next_decision < 0):
            continue

        # Check for collisions with previously marked areas
        original_end = original_position + step_size
        current_area = set(range(original_position, original_end))
        if not (current_area & marked_areas):  # No collision
            marked_areas.update(current_area)
            sure_signals.append(sure_signal_idx_to_append)

    return sure_signals

def load_config(model_name: str):
    with open('config.json', 'r') as f:
        config = json.load(f)
        if model_name in config:
            return config[model_name]
        else:
            raise ValueError(f"No configuration found for model: {model_name}")

def parse_pulsarnet_output_file(log_file_loc):

    # Load the text from the file
    with open(log_file_loc, 'r') as file:
        data = file.read()

    # Regular expressions for various patterns
    time_taken_pattern = re.compile(r'Time taken (.*?): (\d+(\.\d+)?)s')
    processing_file_pattern = re.compile(r'Processing file: (.*?)\n')
    other_info_pattern = re.compile(r'^(.*?): (.*)$', re.MULTILINE)
    table_pattern = re.compile(r'^(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+\.\d+)', re.MULTILINE)

    # Extracting various information
    time_taken_matches = time_taken_pattern.findall(data)
    processing_file = processing_file_pattern.search(data).group(1)
    other_info_matches = other_info_pattern.findall(data)
    table_matches = table_pattern.findall(data)

    # Extracting table values to lists
    _, f_values, periods, z_values, start_index, confidence_values = zip(*[(int(m[0]), float(m[1]), float(m[2]), float(m[3]), int(m[4]), float(m[5])) for m in table_matches])

    times_dict = {match[0]: float(match[1]) for match in time_taken_matches}
    obs_id = processing_file.split('/')[-1].split('.')[0]
    other_info_dict = {match[0]: match[1] for match in other_info_matches}
    return obs_id, processing_file, times_dict, other_info_dict, np.array(f_values), np.array(periods), np.array(z_values),np.array(start_index), np.array(confidence_values)
