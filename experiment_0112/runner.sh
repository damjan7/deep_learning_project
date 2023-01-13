#!/usr/bin/sh
#BSUB -J "DL_CPE"
#BSUB -n 1
#BSUB sbatch --gpus=1
#BSUB -W 4:00  

source /cluster/apps/local/env2lmod.sh
module load gcc/6.3.0 python_gpu/3.8.5
python run_experiment.py
