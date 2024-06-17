#!/bin/bash
 
#SBATCH --job-name=train_disease
#SBATCH --output=train_disease-wandb-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=train_disease-wandb-%A.err  # Standard error of the script
#SBATCH --time=7-00:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=16  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=20G  # Memory in GB (Don't use more than 48GB per GPU unless you absolutely need it and know what you are doing)
#SBATCH --exclude=leto
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cobainnguyen@gmail.com
# run the program
python chexpert.disease.nonli.wandb.cook.py --config /vol/aimspace/users/ngq/Towards-Fair-and-Private-AI-in-Chest-Radiograph-Analysis/config/sweep_non_private_maximize_all_val_AUC_minus_FPR.yml