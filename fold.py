'''
Fold a candidate from PulsarNet.txt file
'''
import os
import argparse
import glob
import re
import time
from settings import G, c, M_SUN
from utils import load_config, parse_pulsarnet_output_file

# Setting up argparse
parser = argparse.ArgumentParser(description="Script parameters")
parser.add_argument("PulsarNet_file", type=str, help="Path to the PulsarNet candidate file")
parser.add_argument("-M", "--model", type=str, default="modelA", help="Model to use for inference")
parser.add_argument("-c", "--cand", nargs='+', type=int, required=True, help="Candidate number(s) to fold")
parser.add_argument("--only_cmd", action='store_true', help="Only print the command to be executed")
parser.add_argument("-o", "--output_label", type=str, default='', help="Output label for the results.")

# Function definitions
def a_from_z(z, T_obs, h, P_s):
    '''
    Calculate acceleration a [m/s^2] from drift z
    z: number of bins drifted
    T_obs: observation time in hours
    h: harmonic number
    P_s: spin period in seconds
    '''
    T_obs = T_obs * 3600 
    return z * P_s * c / (T_obs**2 * h)

def a_to_pdot(P_s, acc_ms2):
    '''
    Calculate period derivative from acceleration
    P_s: spin period in seconds
    acc_ms2: acceleration in m/s^2
    '''
    return P_s * acc_ms2 /c

def calculate_presto_fold_p_neg(p,pd,T_obs):
    '''
    p: period in seconds
    pd: period derivative in seconds/second
    T_obs: observation time in hours
    '''
    T_obs = T_obs*3600
    return p - pd*T_obs/2  

# Edit: Replace all print statements with myexecute 
def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo '%s'"%cmd)
    os.system(cmd)

def main():
    time_start = time.time()
    args = parser.parse_args()

    # Load configuration
    cfg_file_path = 'model_settings.cfg'
    parsed_data = load_config(cfg_file_path)
    sing_img = parsed_data['General']['presto_singularity_image']
    sing_prefix = 'singularity exec -H $HOME:/home1 -B /hercules:/hercules/ ' + sing_img + ' '
    
    root_dir = args.output_label
    cur_dir = './'

    output_label, dat_file, times_dict, other_info_dict, f_array, p_array, z_array, start_index_array, confidence_array = parse_pulsarnet_output_file(args.PulsarNet_file)
    first_iteration = True
    for cand in args.cand:
        cand_index = cand - 1
        assert cand_index >= 0

        p_pred = p_array[cand_index] / 1000
        z = z_array[cand_index]
        T_obs = float(other_info_dict['T_obs'])
        a_pred_abs = a_from_z(z, T_obs / 60, 1, p_pred)
        pd_pred_abs = a_to_pdot(p_pred, a_pred_abs)
        p_fold_pos_z = calculate_presto_fold_p_neg(p_pred, pd_pred_abs, T_obs / 60)
        p_fold_neg_z = calculate_presto_fold_p_neg(p_pred, -pd_pred_abs, T_obs / 60)

        prepfold_out_dir_pos = f'{root_dir}{output_label}_pos_Cand{cand}'
        prepfold_out_dir_neg = f'{root_dir}{output_label}_neg_Cand{cand}'
        prep_base_dir = os.path.dirname(prepfold_out_dir_pos)
        
        #Edit: this is not the most elegant way to do this, but it works
        temp_name_pos = os.path.join(cur_dir, os.path.basename(prepfold_out_dir_pos))
        temp_name_neg = os.path.join(cur_dir, os.path.basename(prepfold_out_dir_neg))

        # Execute prepfold commands for each candidate
        if args.only_cmd:
            print(f'########## Candidate {cand} ##########')
            print(f'prepfold -topo -coarse -p {p_fold_pos_z} -pd {pd_pred_abs} -o {prepfold_out_dir_pos} {dat_file}')
            print(f'prepfold -topo -coarse -p {p_fold_neg_z} -pd {-pd_pred_abs} -o {prepfold_out_dir_neg} {dat_file}')
        else:
            myexecute(sing_prefix + f'prepfold -topo -coarse -p {p_fold_pos_z} -pd {pd_pred_abs} -o {temp_name_pos} -noxwin {dat_file}')
            myexecute(sing_prefix + f'prepfold -topo -coarse -p {p_fold_neg_z} -pd {-pd_pred_abs} -o {temp_name_neg} -noxwin {dat_file}')

            # Handle files for each candidate
            temp_name_pos += '*'
            temp_name_neg += '*'
            temp_name_pos_file_list = glob.glob(temp_name_pos)
            temp_name_neg_file_list = glob.glob(temp_name_neg)

            # Rsync and delete local files
            for file in temp_name_pos_file_list:
                os.system(f'rsync -Pav {file} {prep_base_dir}')
                os.system(f'rm {file}')
            for file in temp_name_neg_file_list:
                os.system(f'rsync -Pav {file} {prep_base_dir}')
                os.system(f'rm {file}')

        pos_file = f'{root_dir}{output_label}_pos_Cand{cand}*pfd.bestprof'
        neg_file = f'{root_dir}{output_label}_neg_Cand{cand}*pfd.bestprof'
        
        pos_sigma, neg_sigma = 0, 0

        if os.path.exists(pos_file):
            with open(pos_file, 'r') as file:
                lines = file.readlines()
                line = lines[13]  # Assuming the sigma value is on line 14
                match = re.search(r'~(\d+.\d+) sigma', line)
                if match:
                    pos_sigma = float(match.group(1))

        if os.path.exists(neg_file):
            with open(neg_file, 'r') as file:
                lines = file.readlines()
                line = lines[13]  # Assuming the sigma value is on line 14
                match = re.search(r'~(\d+.\d+) sigma', line)
                if match:
                    neg_sigma = float(match.group(1))

        best = 'pos' if pos_sigma >= neg_sigma else 'neg'
        best_sigma = pos_sigma if pos_sigma >= neg_sigma else neg_sigma
        
        mode = 'w' if first_iteration else 'a'
        # Writing the best result of each candidate to a file
        with open(f'{root_dir}{output_label}_Candidates.txt', mode) as file:
            if first_iteration:
                first_iteration = False
            file.write(f'Candidate:{cand}\n')
            file.write(f'{"Parameter":<35}{"Value":<15}\n')
            file.write(f'{"-"*35:<35}{"-"*15:<15}\n')
            
            file.write(f'{"predicted_p_middle[ms]:":<35}{p_pred*1000:.10f}\n')
            if best == 'pos':
                file.write(f'{"predicted_p_fold_pos_z[ms]:":<35}{p_fold_pos_z*1000:.10f}\n')
                file.write(f'{"predicted_pdot[s/s]:":<35}{pd_pred_abs:.10e}\n')
                file.write(f'{"predicted_z[index]:":<35}{z}\n')
            else:
                file.write(f'{"predicted_p_fold_neg_z[ms]:":<35}{p_fold_neg_z*1000:.10f}\n')
                file.write(f'{"predicted_pdot[s/s]:":<35}{-pd_pred_abs:.10e}\n')
                file.write(f'{"predicted_z[index]:":<35}{-z}\n')

            file.write(f'{"predicted_sigma_levels:":<35}{best_sigma}\n')
            file.write(f'{"predicted_pos_or_neg_accel_sign:":<35}{best}\n')
            file.write(f'{"-"*50}\n\n')


    #write values in a text file
    with open(f'{root_dir}{output_label}_Candidates.txt', 'a') as file:
        file.write(f'Total elapsed time [s]: {time.time() - time_start:.2f}\n')

if __name__ == "__main__":
    main()