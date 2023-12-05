'''
Fold a candidate from PulsarNet.txt file
'''
import os
from settings import G,c,M_SUN
from utils import load_config, parse_pulsarnet_output_file
import argparse
import glob
import re
import time
import numpy as np

# Setting up argparse
parser = argparse.ArgumentParser(description="Script parameters")

parser.add_argument("PulsarNet_file", type=str, help="Path to the PulsarNet candidate file")
parser.add_argument("--model", type=str, default="modelA", help="Model to use for inference")
parser.add_argument("--cand",type=int,default=1,help="Candidate number to fold")
parser.add_argument("--only_cmd",action='store_true',help="Only print the command to be executed")


def a_from_z(z,T_obs,h,P_s):
    T_obs = T_obs*3600
    return z*P_s*c/(T_obs**2*h)

def a_to_pdot(P_s, acc_ms2):
    return P_s * acc_ms2 /c

def calculate_presto_fold_p_neg(p,pd,T_obs):
    '''
    p: period in seconds
    pd: period derivative in seconds/second
    T_obs: observation time in hours
    '''
    T_obs = T_obs*3600
    return p - pd*T_obs/2  

def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo '%s'"%cmd)
    os.system(cmd)

def main():
    time_start = time.time()
    args = parser.parse_args()
    model = args.model
    config = load_config(model)
    sing_prefix = config['presto_fold_singularity_prefix']
    cand = args.cand - 1 # Python indexing starts from 0
    assert cand >= 0 

#edit this function since pulsarnet also outputs the period now
    output_label, dat_file, times_dict, other_info_dict, f_array, p_array, z_array,start_index_array, confidence_array = parse_pulsarnet_output_file(args.PulsarNet_file)
    p_pred = p_array[cand]/1000
    z = z_array[cand]
    T_obs = float(other_info_dict['T_obs'])
    a_pred_abs = a_from_z(z,T_obs/60,1,p_pred)
    pd_pred_abs = a_to_pdot(p_pred,a_pred_abs)
    p_fold_pos_z = calculate_presto_fold_p_neg(p_pred,pd_pred_abs,T_obs/60)
    p_fold_neg_z = calculate_presto_fold_p_neg(p_pred,-pd_pred_abs,T_obs/60)

    #sing_prefix = 'singularity exec -H $HOME:/home1 -B /hercules:/hercules/  /hercules/scratch/atya/compare_pulsar_search_algorithms.simg '
    if args.only_cmd:
        print(f'prepfold -topo -coarse -p {p_fold_pos_z} -pd {pd_pred_abs} -o output/{output_label}_pos -noxwin {dat_file}')
        print(f'prepfold -topo -coarse -p {p_fold_neg_z} -pd {-pd_pred_abs} -o output/{output_label}_neg -noxwin {dat_file}')
        return

    myexecute(sing_prefix+f'prepfold -topo -coarse -p {p_fold_pos_z} -pd {pd_pred_abs} -o output/{output_label}_pos -noxwin {dat_file}')
    myexecute(sing_prefix+f'prepfold -topo -coarse -p {p_fold_neg_z} -pd {-pd_pred_abs} -o output/{output_label}_neg -noxwin {dat_file}')

    files_list_temp = glob.glob(f'output/{output_label}*pfd.bestprof')
    sigma_levels = []

    for i in range(len(files_list_temp)):
        file_name_temp = files_list_temp[i]
        with open(file_name_temp, 'r') as file:
            lines = file.readlines()
            line = lines[13]  # Since we count from 0, the 10th line is at index 9
            match = re.search(r'~(\d+.\d+) sigma', line)
            if match:
                sigma_levels.append(float(match.group(1)))

    pos_or_neg_ind = np.argmax(sigma_levels)
    if 'pos' in files_list_temp[pos_or_neg_ind]:
        pos_or_neg = 'pos'
    else:
        pos_or_neg = 'neg'

    #write values in a text file
    with open(f'output/{output_label}_predictions_Cand{cand+1}.txt','w') as file:
        file.write(f'predicted_p_middle[ms]: {p_pred*1000:.10f}\n')
        file.write(f'predicted_a_magnitude[m/s^2]: {a_pred_abs:.5f}\n')
        file.write(f'predicted_pdot_magnitude[s/s]: {format(pd_pred_abs, "e")}\n')
        file.write(f'predicted_p_fold_pos_z[ms]: {p_fold_pos_z*1000:.10f}\n')
        file.write(f'predicted_p_fold_neg_z[ms]: {p_fold_neg_z*1000:.10f}\n')
        file.write(f'predicted_sigma_levels: {sigma_levels}\n')
        file.write(f'predicted_pos_or_neg_accel_sign: {pos_or_neg}\n')
        file.write(f'predicted_z[index]: {z}\n')
        file.write(f'time_taken[s]: {time.time() - time_start:.5f}\n')

if __name__ == "__main__":
    main()