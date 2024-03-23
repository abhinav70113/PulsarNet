"""
utils.py

This module offers utility functions that can be reused throughout the project.
It provides functionalities to extract overlapping windows from an array, compute features, and filter predictions.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
import json
import re
import configparser

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

def filter_chunks_by_decision_value(chunks, classifier, scaler):
    """ Filter chunks based on classifier decision value. """
    features = np.apply_along_axis(compute_features, 1, chunks)
    standardized_features = scaler.transform(features)
    decision_values = classifier.decision_function(standardized_features)
    positive_indices = np.where(decision_values > 0)[0]
    return chunks[positive_indices], decision_values[positive_indices], positive_indices

def find_non_overlapping_chunks(chunk_infos, tolerance):
    """ Identify and filter out consecutive chunks based on start frequency index. """
    sorted_chunks = sorted(chunk_infos, key=lambda x: x['start_freq_ind_chunk'])
    filtered_chunks = []
    
    for i in range(len(sorted_chunks)):
        if i == 0 or sorted_chunks[i]['start_freq_ind_chunk'] - sorted_chunks[i - 1]['start_freq_ind_chunk'] >= tolerance:
            filtered_chunks.append(sorted_chunks[i])
        elif sorted_chunks[i]['confidence'] > sorted_chunks[i - 1]['confidence']:
            filtered_chunks[-1] = sorted_chunks[i]

    return filtered_chunks

def parse_inf_file(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and split the line on '=' if it contains it
            if '=' in line:
                key, value = line.split('=', 1)
                # Further processing to clean up the key and value
                key = key.strip()
                value = value.split("#")[0].strip()  # Remove any comments
                # Add to the dictionary
                data[key] = value
    return data

def load_config(cfg_file_path):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(cfg_file_path)

    # Create a dictionary to hold the parsed data
    config_data = {}

    # Iterate over the sections and options in the file
    for section in config.sections():
        config_data[section] = {}
        for option in config.options(section):
            config_data[section][option] = config.get(section, option)

    return config_data

def parse_pulsarnet_output_file(log_file_loc):
    '''
    Parses the PulsarNet output file and returns various information.

    Parameters:
    - log_file_loc (str): Location of the PulsarNet output file.
    
    Returns:
    - info_dict (dict): Dictionary containing meta information in the form of a dictionary.
        - Processing_file (str): Processing file.
        - time_res (float): Time resolution.
        - T_obs (float): Observation time.
        - freq_res (float): Frequency resolution.
        - fft_size (int): FFT size.
        - Number of signals found (int): Number of signals found.
        - Time total elapsed (float): Total time elapsed.
        - File processing (float): File processing time.
        - Classifier stage (float): Classifier stage time.
        - Predictor stage (float): Predictor stage time.
        - Model loading time (float): Model loading time.
    - f_values (np.ndarray): Array containing frequency values.
    - periods (np.ndarray): Array containing period values.
    - z_values (np.ndarray): Array containing z values.
    - start_index (np.ndarray): Array containing start index values.
    - confidence_values (np.ndarray): Array containing confidence values.
    '''
    # Load the text from the file
    with open(log_file_loc, 'r') as file:
        data = file.read()

    # Processing file: obs4024BH.dat
    # time_res: 6.4e-05
    # T_obs: 17.895697066666667
    # freq_res: 0.0009313225746154785
    # fft_size: 8388609
    # Number of signals found: 408
    # Time total elapsed: 8.34s
    # File processing: 1.23s
    # Classifier stage: 4.87s
    # Predictor stage: 0.83s
    # Model loading time: 1.34s

    # Regular expressions for various patterns
    #time_taken_pattern = re.compile(r'Time taken (.*?): (\d+(\.\d+)?)s')
    #processing_file_pattern = re.compile(r'Processing file: (.*?)\n')
    other_info_pattern = re.compile(r'^(.*?): (.*)$', re.MULTILINE)
    table_pattern = re.compile(r'^(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+\.\d+)', re.MULTILINE)

    # Extracting various information
    #time_taken_matches = time_taken_pattern.findall(data)
    #processing_file = processing_file_pattern.search(data).group(1)
    # If processing file ending is .fft, then replace it with .dat, because we only fold .dat files
    # if processing_file.endswith('.fft'):
    #     processing_file = processing_file[:-4] + '.dat'
    other_info_matches = other_info_pattern.findall(data)
    table_matches = table_pattern.findall(data)

    # Extracting table values to lists
    _, f_values, periods, z_values, start_index, confidence_values = zip(*[(int(m[0]), float(m[1]), float(m[2]), float(m[3]), int(m[4]), float(m[5])) for m in table_matches])

    #times_dict = {match[0]: float(match[1]) for match in time_taken_matches}
    #obs_id = processing_file.split('/')[-1].split('.')[0]
    other_info_dict = {match[0]: match[1] for match in other_info_matches}
    #return obs_id, processing_file, times_dict, other_info_dict, np.array(f_values), np.array(periods), np.array(z_values),np.array(start_index), np.array(confidence_values)
    return other_info_dict, np.array(f_values), np.array(periods), np.array(z_values),np.array(start_index), np.array(confidence_values)
