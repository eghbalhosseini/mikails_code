#!/bin/bash
#SBATCH --job-name=matlabtest
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --mem=5gb
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --output=/om/user/ehoseini/MatlabFolder/matlabtest_%A_%a.log
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --array=1
#SBATCH --ntask=1
pwd; hostname; date

set -x
export VAR=`bc -l <<< "$SLURM_ARRAY_TASK_ID"`

cd /om/user/ehoseinil/MatlabFolder/

module add mit/matlab/2018a
matlab -nodisplay -nodesktop -r " addpath(genpath('/om/ehoseini/MatLabFolder/'));grid_dynamics($VAR,'test.mat', .5, 0,1);  quit"

date
