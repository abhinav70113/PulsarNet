import sys
import numpy as np
import glob
import tensorflow as tf
from numpy.lib.stride_tricks import as_strided
import time
import joblib
import json
import argparse
import os
from settings import M_SUN, G, c
from file_processing import process_dat, process_fft
from utils import overlapping_windows, load_config, parse_inf_file, find_non_overlapping_chunks, filter_chunks_by_decision_value
import subprocess
# Setting up argparse
parser = argparse.ArgumentParser(description="Script parameters")

parser.add_argument("dat_or_fft_file", type=str, help="Path to the dat or fft file")
parser.add_argument("-s","--freq_start", type=float, default=40, help="Frequency start in Hz")
parser.add_argument("-e","--freq_end", type=float, default=1000, help="Frequency end in Hz")
parser.add_argument("-M","--model", type=str, default="modelA", help="Model to use for inference")
# make this required with no default
parser.add_argument("-o", "--output_label", type=str, required=True, help="Output label for the results.")
# Add a check_gpu flag
parser.add_argument("--check_gpu", action='store_true', help="Only check if GPU is available for use")

def main():
    args = parser.parse_args()

    if args.check_gpu:
        subprocess.run(["echo", "TensorFlow version:", tf.__version__])
        subprocess.run(["echo", "Is GPU available:", tf.test.is_gpu_available()])
        subprocess.run(["echo", "Visible devices:", tf.config.list_physical_devices()])
        subprocess.run(["echo", "Run the program without the --check_gpu flag to continue"])
        # print("TensorFlow version:", tf.__version__)
        # print("Is GPU available:", tf.test.is_gpu_available())
        # print("Visible devices:", tf.config.list_physical_devices())
        # print('Run the program without the --check_gpu flag to continue')
        sys.exit()

    # Edit: Check if this logger_info string method might be bad for RAM since it has to keep storing everything in memory, problematic for long candidate lists
    # Edit: Add this under verbosity levels
    #tf.debugging.set_log_device_placement(True)
    #physical_devices = tf.config.list_physical_devices()
    # Edit: Add this under verbosity levels and logger
    # Edit: See if all print statements are to be left in or removed
    # echo instead of print
    #subprocess.run(["echo", "Available devices:", str(physical_devices)])
    #print("Available devices:", physical_devices)
    #dat_or_fft_file = f'/hercules/results/atya/BinaryML/sims/{args.run}/dat_inf_files/{args.dat_or_fft_file}'
    

    dat_or_fft_file = args.dat_or_fft_file

    freq_start = args.freq_start
    freq_end = args.freq_end
    model_name = args.model
    output_dir = args.output_label

    # Get the directory where this code is saved
    code_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(["echo", f"Code directory: {code_dir}"])
    #print(f"Code directory: {code_dir}")
    # Edit: Maybe don't harcode the config file path 
    cfg_file_path = os.path.join(code_dir,'model_settings.cfg')
    parsed_data = load_config(cfg_file_path)
    config = parsed_data[model_name]

    # Parse the inf file name to get meta data
    inf_file_name = dat_or_fft_file[:-3] + 'inf'

    # Check if the inf file exists
    if not os.path.exists(inf_file_name):
        subprocess.run(["echo", f"Error: {inf_file_name} does not exist"])
        #print(f"Error: {inf_file_name} does not exist")
        sys.exit()
    # Check if the dat or fft file exists
    if not os.path.exists(dat_or_fft_file):
        subprocess.run(["echo", f"Error: {dat_or_fft_file} does not exist"])
        #print(f"Error: {dat_or_fft_file} does not exist")
        sys.exit()

    parsed_inf_file = parse_inf_file(inf_file_name)
    time_res = np.float64(parsed_inf_file['Width of each time series bin (sec)'])
    subprocess.run(["echo", f"Time resolution: {time_res}"])
    #print(f"Time resolution: {time_res}")

    logger_info = f'time_res: {time_res}\n'

    model_loading_start_time = time.time()
    scaler = joblib.load(os.path.join(code_dir,config["scaler"]))
    clf    = joblib.load(os.path.join(code_dir,config["classifier"]))
    # model_regressor_z = tf.keras.models.load_model('/hercules/scratch/atya/PulsarNet/models/z_predict_attention_6445575_runBF_checkpoint.h5')
    # model_regressor_f = tf.keras.models.load_model('/hercules/scratch/atya/PulsarNet/models/f_predict_LSTM_6446470_runBF_checkpoint.h5')
    #model_regressor_z = tf.keras.models.load_model('/hercules/scratch/atya/BinaryML/hyperparameter_tuning/attention_z/tuner_predict_attention_z_9471055_619_checkpoint.h5')
    #model_regressor_f = tf.keras.models.load_model('/hercules/scratch/atya/BinaryML/models/tuner_predict_cnn_f_10354172_200_checkpoint.h5')

    model_regressor_z = tf.keras.models.load_model(os.path.join(code_dir,config["z_model"]))
    model_regressor_f = tf.keras.models.load_model(os.path.join(code_dir,config["f_model"]))

    
    max_z = np.float64(config["max_z"])
    max_f = np.float64(config["max_f"])
    step_size = np.int32(config["step_size"])
    chunk_size = np.int32(config["chunk_size"])
    raw_data_normalization_for_z = bool(config["raw_data_normalization_for_z"])
    raw_data_normalization_for_f = bool(config["raw_data_normalization_for_f"])

    model_loading_end_time = time.time()
    logger_info += f'Time taken to load models: {(model_loading_end_time - model_loading_start_time):.2f}s\n'
    subprocess.run(["echo", f"Time taken to load models: {(model_loading_end_time - model_loading_start_time):.2f}s"])
    subprocess.run(["echo", f"Models loaded sucessfully"])
    #print("Models loaded sucessfully")

    file_process_time_start = time.time()
    # Edit: this is weird because if the fft file is passed instead, the prefpold will be formed using the fft file name and crash therefore, since it can only take
    logger_info+=f'Processing file: {dat_or_fft_file}\n'
    subprocess.run(["echo", f"Processing file: {dat_or_fft_file}"])

    out_file_list = dat_or_fft_file.split('/')
    file_name = out_file_list[-1]
    root_dir = output_dir
    extension = file_name.split('.')[-1]
    #ind = int(sys.argv[1])

    if extension == 'dat':
        power = process_dat(dat_or_fft_file)
        file_process_end_time = time.time()
        logger_info+=f'Time taken to do fft and normalize: {(file_process_end_time - file_process_time_start):.2f}s\n'
        subprocess.run(["echo", f"Time taken to do fft and normalize: {(file_process_end_time - file_process_time_start):.2f}s"])
    elif extension == 'fft':
        power = process_fft(dat_or_fft_file)
        file_process_end_time = time.time()
        logger_info+=f'Time taken to load fft and normalize: {(file_process_end_time - file_process_time_start):.2f}s\n'
        subprocess.run(["echo", f"Time taken to load fft and normalize: {(file_process_end_time - file_process_time_start):.2f}s"])
    else:
        subprocess.run(["echo", "Invalid file extension"])
        #print('Invalid file extension')
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

    subprocess.run(["echo", "FFT processsed sucessfully"])
    #print("FFT processsed sucessfully")
    freq_ind_start = np.argmin(np.abs(freq_axis - freq_start)) - chunk_size
    if freq_ind_start < 0:
        freq_ind_start = 0
    freq_ind_end = np.argmin(np.abs(freq_axis - freq_end)) + chunk_size

    roll_number = 0
    num_chunks = (freq_ind_end - freq_ind_start) // step_size - 1
    rolled_freq_start_ind = freq_ind_start + roll_number
    rolled_freq_end_ind = freq_ind_start + step_size*(num_chunks + 1) + roll_number
    X_test_freq = power[rolled_freq_start_ind:rolled_freq_end_ind]
    X_test_freq_chunks = overlapping_windows(X_test_freq, step_size, chunk_size)
    filtered_chunks, decision_values, positive_indices = filter_chunks_by_decision_value(X_test_freq_chunks, clf, scaler)

    # Edit: all of the hardcoded values corresponding to the new classifier should be put in cfg file
    temp_step_size = config["tolerance"]
    temp_roll_number_list = np.arange(temp_step_size,chunk_size//2,temp_step_size,dtype=int)
    num_temp_chunks = 4
    chunk_infos = []

    # Further chunk processing
    for start_multiple in range(num_temp_chunks):
        for temp_roll_number in temp_roll_number_list:
            chunk_infos.extend([{'start_freq_ind_chunk':(loc + 1) * step_size + freq_ind_start + roll_number - chunk_size + temp_roll_number + start_multiple*step_size , 'confidence': decision_values[i]} for i, loc in enumerate(positive_indices)])
    print('chunk infos',len(chunk_infos))
    non_overlapping_chunks = find_non_overlapping_chunks(chunk_infos, tolerance=temp_step_size)  # NO_chunks
    non_overlapping_X_array = np.array([power[chunk['start_freq_ind_chunk']:chunk['start_freq_ind_chunk'] + step_size] for chunk in non_overlapping_chunks])
    print('non overlapping X array',len(non_overlapping_X_array))

    scaler_50 = joblib.load(os.path.join(code_dir,config["scaler_50"]))
    clf_50    = joblib.load(os.path.join(code_dir,config["classifier_50"]))
    
    NO_X_array, NO_decision_values, NO_positive_indices = filter_chunks_by_decision_value(non_overlapping_X_array, clf_50, scaler_50)
    NO_chunk_infos = [{'start_freq_ind_chunk':non_overlapping_chunks[i]['start_freq_ind_chunk'],'confidence_50':NO_decision_values[ind],'pos_index':ind} for ind,i in enumerate(NO_positive_indices)]
    print(len(NO_chunk_infos))
    # Step 1: Sort stored_info items based on confidence
    sorted_chunks = sorted(NO_chunk_infos, key=lambda x: x['confidence_50'], reverse=True)
    marked_areas = set()
    final_results = []

    for dict in sorted_chunks:
        start_freq_ind_chunk = dict['start_freq_ind_chunk']

        # Step 2: Calculate original location in unrolled 'power' spectrum
        original_start = start_freq_ind_chunk
        original_end = original_start + step_size  # Assuming chunk_size is the length of each chunk

        # Step 3: Check for collisions with previously marked areas
        current_area = set(range(original_start, original_end))
        if not (current_area & marked_areas):  # No collision
            marked_areas.update(current_area)
            final_results.append(dict)

    subprocess.run(["echo", f"Classifier final results: Number of Candidates {len(final_results)}"]) 

    # if the final result has more than 5000 candidates, then truncate the rest
    if len(final_results) > 5000:
        final_results = final_results[:5000]

    final_pos_indices = [dict['pos_index'] for dict in final_results]
    start_indices = [dict['start_freq_ind_chunk'] for dict in final_results]
    confidences = [dict['confidence_50'] for dict in final_results]
    signals = NO_X_array[final_pos_indices]

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

    subprocess.run(["echo", f"Number of signals found: {len(final_results)}"])
    #print(f"Number of signals found: {len(final_results)}")
    subprocess.run(["echo", f"Time taken for search: {(search_time_stop - search_time_start):.2f}s"])
    #print(f"Time taken for search: {(search_time_stop - search_time_start):.2f}s")
    subprocess.run(["echo", '-'*75])
    #print('-'*75)

    subprocess.run(["echo", "Regressor sucessfully processed all chunks. Writing results to the file"])
    #print("Regressor:")
    print("Regressor sucessfully processed all chunks. Writing results to the file")

    # Create or overwrite the external file
    with open(f'{root_dir}_PulsarNet.txt', 'w') as file:
        
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
