"""
file_processing.py

This module provides functionalities to process .dat and .fft files.
It contains methods to extract and normalize the power values from the respective file types.
"""

import numpy as np

def process_dat(dat_file: str) -> np.ndarray:
    """
    Process the provided .dat file and return its normalized power.

    Parameters:
    - dat_file (str): Path to the .dat file.

    Returns:
    - np.ndarray: Normalized power values from the processed .dat file.
    """
    dat = np.fromfile(dat_file, dtype=np.float32)
    dat = (dat - np.mean(dat)) / np.std(dat)
    fft = np.fft.rfft(dat)
    power = np.abs(fft)**2
    power_norm = (power - np.mean(power))/np.std(power)
    return power_norm

def process_fft(fft_file: str) -> np.ndarray:
    """
    Process the provided .fft file and return its normalized power.

    Parameters:
    - fft_file (str): Path to the .fft file.

    Returns:
    - np.ndarray: Normalized power values from the processed .fft file.
    """
    fft_packed = np.fromfile(fft_file, dtype=np.complex64)
    N_over_2 = len(fft_packed)
    # Create an empty array for the unpacked FFT
    fft_unpacked = np.zeros(N_over_2 + 1, dtype=np.complex64)
    # Place the DC component (real part of the zeroth bin) at the beginning
    # set it 0 because we want the mean of the data series to be 0
    fft_unpacked[0] = 0.0 
    # Place the Nyquist component (imaginary part of the zeroth bin) at the end
    fft_unpacked[-1] = fft_packed[0].imag
    # Copy over the remaining data points
    fft_unpacked[1:-1] = fft_packed[1:]
    # Account for the normalization of the time series
    sigma_fact = (2*np.sum(np.abs(fft_unpacked)**2) - np.abs(fft_unpacked[-1])**2)/(N_over_2*2)
    power = np.abs(fft_unpacked)**2/sigma_fact
    # Normalize the fft
    power_norm = (power - np.mean(power))/np.std(power)
    return power_norm
