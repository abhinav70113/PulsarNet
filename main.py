import numpy as np
import glob
import tensorflow as tf
from numpy.lib.stride_tricks import as_strided
import sys
import time
import joblib
import json
import argparse
import os
from settings import M_SUN, G, c
from file_processing import process_dat, process_fft
from utils import overlapping_windows, compute_features, filter_predictions, load_config

# Setting up argparse
parser = argparse.ArgumentParser(description="Script parameters")

parser.add_argument("dat_or_fft_file", type=str, help="Path to the dat or fft file")

parser.add_argument("--freq_start", type=float, default=40, help="Frequency start in Hz")
parser.add_argument("--freq_end", type=float, default=1000, help="Frequency end in Hz")
parser.add_argument("--time_res", type=float, default=64e-6, help="Time resolution in seconds")
parser.add_argument("--model", type=str, default="modelA", help="Model to use for inference")

def main():
    args = parser.parse_args()

    #dat_or_fft_file = f'/hercules/results/atya/BinaryML/sims/{args.run}/dat_inf_files/{args.dat_or_fft_file}'
    dat_or_fft_file = args.dat_or_fft_file

    freq_start = args.freq_start
    freq_end = args.freq_end
    time_res = args.time_res
    model_name = args.model
    config = load_config(model_name)

    logger_info = f'time_res: {time_res}\n'

    model_loading_start_time = time.time()
    scaler = joblib.load(config["scaler"])
    clf = joblib.load(config["classifier"])
    # model_regressor_z = tf.keras.models.load_model('/hercules/scratch/atya/PulsarNet/models/z_predict_attention_6445575_runBF_checkpoint.h5')
    # model_regressor_f = tf.keras.models.load_model('/hercules/scratch/atya/PulsarNet/models/f_predict_LSTM_6446470_runBF_checkpoint.h5')
    #model_regressor_z = tf.keras.models.load_model('/hercules/scratch/atya/BinaryML/hyperparameter_tuning/attention_z/tuner_predict_attention_z_9471055_619_checkpoint.h5')
    #model_regressor_f = tf.keras.models.load_model('/hercules/scratch/atya/BinaryML/models/tuner_predict_cnn_f_10354172_200_checkpoint.h5')

    model_regressor_z = tf.keras.models.load_model(config["z_model"])
    model_regressor_f = tf.keras.models.load_model(config["f_model"])

    
    max_z = config["max_z"]
    max_f = config["max_f"]
    step_size = config["step_size"]
    chunk_size = config["chunk_size"]
    raw_data_normalization_for_z = bool(config["raw_data_normalization_for_z"])
    raw_data_normalization_for_f = bool(config["raw_data_normalization_for_f"])

    model_loading_end_time = time.time()
    logger_info += f'Time taken to load models: {(model_loading_end_time - model_loading_start_time):.2f}s\n'
    print("Models loaded sucessfully")

    file_process_time_start = time.time()
    logger_info+=f'Processing file: {dat_or_fft_file}\n'

    out_file = dat_or_fft_file.split('/')[-1]
    #root_dir = '/'.join(dat_or_fft_file.split('/')[:-1])
    root_dir = os.getcwd()
    output_label, extension = out_file.split('.')
    #ind = int(sys.argv[1])

    if extension == 'dat':
        power = process_dat(dat_or_fft_file)
        file_process_end_time = time.time()
        logger_info+=f'Time taken to do fft and normalize: {(file_process_end_time - file_process_time_start):.2f}s\n'
    elif extension == 'fft':
        power = process_fft(dat_or_fft_file)
        file_process_end_time = time.time()
        logger_info+=f'Time taken to load fft and normalize: {(file_process_end_time - file_process_time_start):.2f}s\n'
    else:
        print('Invalid file extension')
        sys.exit()


    search_time_start = time.time()
    fft_size = len(power)
    dat_size = 2*(fft_size - 1)
    T_obs = (dat_size*time_res)/60 # in minutes is equal to 17.895 minutes
    freq_axis = np.fft.rfftfreq(dat_size, d=time_res)
    freq_res = 1/(T_obs*60)
    logger_info+=f'T_obs: {T_obs}\n'
    logger_info+=f'freq_res: {freq_res}\n'
    logger_info+=f'fft_size: {fft_size}\n'

    print("FFT processsed sucessfully")

    freq_ind_start = np.argmin(np.abs(freq_axis - freq_start)) - chunk_size
    if freq_ind_start < 0:
        freq_ind_start = 0
    freq_ind_end = np.argmin(np.abs(freq_axis - freq_end)) + chunk_size

    #rolled_pred = {}
    stored_info = {}

    roll_list = [0,10,20,30,40]
    num_chunks = (freq_ind_end - freq_ind_start) // step_size - 1
    #print(num_chunks)

    for i, roll_number in enumerate(roll_list):
        rolled_freq_start_ind = freq_ind_start + roll_number
        rolled_freq_end_ind = freq_ind_start + step_size*(num_chunks + 1) + roll_number
        X_test_freq = power[rolled_freq_start_ind:rolled_freq_end_ind]
        X_test_freq_chunks = overlapping_windows(X_test_freq, step_size, chunk_size)
        #print(X_test_freq_chunks.shape)

        X_features = np.apply_along_axis(compute_features, 1, X_test_freq_chunks)
        X_features_std = scaler.transform(X_features)

        decision_values = clf.decision_function(X_features_std)
        filtered_loc_list = filter_predictions(decision_values,step_size,freq_ind_start,roll_number)

        for loc in filtered_loc_list:
            confidence = decision_values[loc]
            # print(ind)
            # print(loc)
            # print(step_size)
            # print(freq_ind_start)
            # print(roll_number)
            # print('-----------------')
            start_freq_ind_chunk = (loc + 1) * step_size + freq_ind_start + roll_number
            
            signal_chunk = X_test_freq_chunks[loc,step_size:]
            
            key = (loc, roll_number)
            stored_info[key] = {
                'confidence': confidence,
                'start_freq_ind_chunk': start_freq_ind_chunk,
                'signal_chunk': signal_chunk
            }

        print(f"Classifier: Roll {roll_number}, Candidates {len(filtered_loc_list)}")

        #rolled_pred[roll_number] = filtered_loc_list
        #print(len(filtered_loc_list))

    # Step 1: Sort stored_info items based on confidence
    sorted_chunks = sorted(stored_info.items(), key=lambda x: x[1]['confidence'], reverse=True)

    marked_areas = set()
    final_results = []

    for key, chunk_info in sorted_chunks:
        roll_number = key[1]  # Extracting roll_number from the tuple key
        start_freq_ind_chunk = chunk_info['start_freq_ind_chunk']

        # Step 2: Calculate original location in unrolled 'power' spectrum
        original_start = start_freq_ind_chunk
        original_end = original_start + step_size  # Assuming chunk_size is the length of each chunk

        # Step 3: Check for collisions with previously marked areas
        current_area = set(range(original_start, original_end))
        if not (current_area & marked_areas):  # No collision
            marked_areas.update(current_area)
            final_results.append(chunk_info)

    print(f"Classifier final results: Number of Candidates {len(final_results)}")
    # Step 4: By now, final_results should have all the chunks that found unique positions in the original spectrum
    signals = np.zeros((len(final_results), step_size))
    start_indices = np.zeros(len(final_results))
    confidences = np.zeros(len(final_results))
    # chunk_numbers = np.zeros(len(final_results))
    # roll_numbers = np.zeros(len(final_results))
    #for i, (chunk_number, (signal_chunk, confidence, start_freq_ind_chunk, roll_number)) in enumerate(final_results.items()):
    for i, result in enumerate(final_results):
        signals[i] = result['signal_chunk']
        start_indices[i] = result['start_freq_ind_chunk']
        confidences[i] = result['confidence']

    signals = signals.reshape((-1,step_size,1)).astype(np.float64)
    signals_normed = signals/np.max(signals,axis=1)[:,None]

    if raw_data_normalization_for_f:
        f = np.clip(model_regressor_f(signals_normed).numpy().reshape(-1),0,max_f)
    else:
        f = np.clip(model_regressor_f(signals).numpy().reshape(-1),0,max_f)

    if raw_data_normalization_for_z:
        z = np.clip(model_regressor_z(signals_normed).numpy().reshape(-1),0,max_z)
    else:
        z = np.clip(model_regressor_z(signals).numpy().reshape(-1),0,max_z)

    logger_info+=f'Number of signals found: {len(final_results)}\n'
    search_time_stop = time.time()
    logger_info+=f'Time taken for search: {(search_time_stop - search_time_start):.2f}s\n\n'
    logger_info+='-'*75 + '\n'

    print("Regressor sucessfully processed all chunks. Writing results to the file")

    # Create or overwrite the external file
    with open(f'{root_dir}/output/{output_label}_PulsarNet.txt', 'w') as file:
        
        # Write logger_info to the file
        file.write(logger_info)
        
        # Write the header for the data
        file.write('Index'.ljust(15))
        file.write('f Value'.ljust(15))
        file.write('period [ms]'.ljust(15))
        file.write('z Value'.ljust(15))
        file.write('Start Index'.ljust(15))
        file.write('Confidence\n')
        file.write('-'*75 + '\n')  
        
        # Write each data entry
        for i in range(len(final_results)):
            f_val = f"{f[i]:.6f}".ljust(15)  # Format to 6 decimal places and left-justify to 15 spaces
            z_val = f"{z[i]:.6f}".ljust(15)  # Same for z values
            s_idx = str(int(start_indices[i])).ljust(15)
            period = 1000/((f[i]+start_indices[i])*freq_res)
            p_val = f"{period:.6f}".ljust(15)
            conf = f"{confidences[i]:.6f}".ljust(15)   # Same for confidences
            
            file.write(f"{i+1}".ljust(15) + f_val + p_val + z_val + s_idx + conf + '\n')
    
    print("Complete.")

if __name__ == "__main__":
    main()