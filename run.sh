#!/bin/bash
#SBATCH -JquietSTAR            # Job name  
#SBATCH -N1 --gres=gpu:H100:8            # Number of nodes and cores per node required  
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=64              # Memory per CPU core, 8 CPUs/GPU 
#SBATCH -t3:00:00                        # Duration of the job (Ex: 1 hour) 
#SBATCH -q ice-gpu
#SBATCH -oReport-%j.out
#SBATCH --output=slurm_out/Report-%A.out

module load anaconda3/2022.05.0.1               # Load module dependencies
conda activate quietSTAR-v2

echo "Running the following command:"
echo $@

srun $@