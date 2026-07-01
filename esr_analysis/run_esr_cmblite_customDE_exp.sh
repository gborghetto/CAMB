#!/bin/bash
###
###
#job name
#SBATCH --job-name=esr_Vphi
#SBATCH --array=0-6
#job stdout file
#SBATCH --output=logs/esr_Vphi_%a.out
#job stderr file
#SBATCH --error=logs/esr_Vphi_%a.err
#maximum job time in D-HH:MM
#SBATCH --time=2-23:59
#number of parallel processes (tasks) you are requesting - maps to MPI processes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#memory per process in MB
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=2362709@swansea.ac.uk     # Where to send mail

# Load required modules
module load anaconda/2023.09
module load compiler/gnu/12/1.0
module load mpi/openmpi/4.1.5
module load texlive/2018

# Activate conda environment
source activate /scratch/s.2362709/swansea/

# Set number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs

N=135
CHUNK=$(( (N + 5) / 7 ))

START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))
if [ $END -gt $N ]; then END=$N; fi

echo "Task $SLURM_ARRAY_TASK_ID: indices $START to $END"
srun --mpi=pmix \
     --ntasks=$SLURM_NTASKS \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     python ESR_camb_full_run_exp_cmblite.py \
         --complex 4 \
         --index1 $START \
         --index2 $END \
         --force
