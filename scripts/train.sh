# defaults
sbatch scripts/run_pace.sh python quiet-star-train.py 

# smaller batch size
sbatch scripts/run_pace.sh python quiet-star-train.py --full_batch_size 4

# eval more, less often
sbatch scripts/run_pace.sh python quiet-star-train.py --eval_every 1000 --eval_steps 500

# increase number of thoughts
sbatch scripts/run_pace.sh python quiet-star-train.py --n_passes_global 4
sbatch scripts/run_pace.sh python quiet-star-train.py --n_passes_global 8

# TODO: other meta prompts