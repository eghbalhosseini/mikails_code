#!/bin/bash
#SBATCH --job-name=matlabtest
#SBATCH --mem=5gb
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --output=/om/user/ehoseini/MatlabFolder/matlabtest_%A_%a.log
#SBATCH --array=1
#SBATCH --ntask=1
pwd; hostname; date

set -x
export VAR=`bc -l <<< "$SLURM_ARRAY_TASK_ID"`

cd /om/user/mikail/MatlabFolder/

module add mit/matlab/2018a
matlab -nodisplay -nodesktop -r "grid_dynamics($VAR,'test.mat', .5, 0,1);  quit"

date
