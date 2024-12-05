# baseline seed test
sbatch scripts/run.sh python quiet-star-train.py --run_name baseline_seed_1-1 --seed 1
sbatch scripts/run.sh python quiet-star-train.py --run_name baseline_seed_1-2 --seed 1
sbatch scripts/run.sh python quiet-star-train.py --run_name baseline_seed_1-3 --seed 1
sbatch scripts/run.sh python quiet-star-train.py --run_name baseline_seed_2-1 --seed 2
sbatch scripts/run.sh python quiet-star-train.py --run_name baseline_seed_2-2 --seed 2
sbatch scripts/run.sh python quiet-star-train.py --run_name baseline_seed_2-3 --seed 2

