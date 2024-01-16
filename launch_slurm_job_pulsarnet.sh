#!/usr/bin/env bash
logs='slurm_job_logs'
mkdir -p $logs

sbatch -p short.q --gres=gpu:1 --err=$logs/test.err --out=$logs/test.out --wrap="/hercules/scratch/vishnu/PulsarNet/run_pulsarnet_node.sh $1"

#sbatch -p short.q --gres=gpu:1 --err=$logs/test1.err --out=$logs/test1.out --wrap="/hercules/scratch/vishnu/PulsarNet/check_gpu.sh"
