#!/bin/bash
#SBATCH -JSlurmPythonExample                    # Job name
#SBATCH --account=gts-agarg35                   # charge account
#SBATCH -N1 --gres=gpu:H100:4                   # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=80G                       # Memory per core
#SBATCH -t8:00:00                               # Duration of the job (8 hours)
#SBATCH -q embers                               # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --output=slurm_out/Report-%A.out

module load anaconda3/2022.05.0.1               # Load module dependencies
conda activate quietSTAR

echo "Running the following command:"
echo $@

srun $@