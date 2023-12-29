#!/bin/bash -l

#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -p course


module load mpi/intel-mpi/2019u5/bin
module load compilers/intel/2019u5



procs=${SLURM_NTASKS}

cores=${SLURM_CPUS_PER_TASK}

export OMP_NUM_THREADS=$cores

mpiicc -qopenmp coordReader.c main-mpi.c ompcInsertion.c ompfInsertion.c ompnAddition.c -std=c99 -lm

mpirun -np $procs ./a.out
