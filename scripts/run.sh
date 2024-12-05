#!/bin/bash
#SBATCH -JSlurmPythonExample                    # Job name
#SBATCH -N1 --gres=gpu:H100:4                   # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=80G                       # Memory per core
#SBATCH --cpus-per-task=32                      # num cpus
#SBATCH -t1:00:00                               # Duration of the job (8 hours)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --output=slurm_out/Report-%A.out

module load anaconda3/2022.05.0.1               # Load module dependencies
conda activate quietSTAR-v2

echo "Running the following command:"
echo $@

srun $@