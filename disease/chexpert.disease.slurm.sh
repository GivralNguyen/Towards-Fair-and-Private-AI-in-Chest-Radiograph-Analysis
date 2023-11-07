#!/bin/sh
 
#SBATCH --job-name=test_slurm
#SBATCH --output=test_slurm-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=test_slurm-%A.err  # Standard error of the script
#SBATCH --time=0-00:10:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=16  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=20G  # Memory in GB (Don't use more than 48GB per GPU unless you absolutely need it and know what you are doing)
#SBATCH --partition=master
#SBATCH --nodelist=prometheus
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cobainnguyen@gmail.com

# # load python module
module load python/anaconda3
module load cuda/11.7
module load cuda/117
# # activate corresponding environment
conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
source activate ngq # If this does not work, try 'source activate ptl'
 
# run the program
python chexpert.disease.py --accelerator gpu