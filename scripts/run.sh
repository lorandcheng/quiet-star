#!/bin/bash
#SBATCH -JSlurmPythonExample                    # Job name
#SBATCH -N1 --gres=gpu:H100:8                   # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=80G                       # Memory per core
#SBATCH --cpus-per-task=64                      # num cpus
#SBATCH -t2:00:00                               # Duration of the job (8 hours)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --output=slurm_out/Report-%A.out

module load anaconda3/2022.05.0.1               # Load module dependencies
conda activate drl

echo "Running the following command:"
echo $@

srun $@