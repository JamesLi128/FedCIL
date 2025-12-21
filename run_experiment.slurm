#!/bin/bash
#SBATCH --job-name=nc100cpr10_cifar10
#SBATCH --partition=gpuq
#SBATCH --qos=gpu                       # required for gpuq on Hopper
#SBATCH --output=/home/hli54/scratch/logs/fcil/cifar10/slurm-log/%j-out.log
#SBATCH --error=/home/hli54/scratch/logs/fcil/cifar10/slurm-log/%j-err.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --time=1-00:00:00

# Load modules (adjust based on your cluster)
# module load python/3.10
module load cuda/12.6

echo "[$(date)] Starting on ${HOSTNAME} (JOB=$SLURM_JOB_ID)"


# Activate conda environment
source ~/miniforge3/bin/activate
conda activate ML  # or your environment name


# Navigate to project
cd ~/Projects/FCIL/models

export PYTHONUNBUFFERED=1

# =============================================
# Example 1: Single run with CIFAR-10 defaults
# =============================================
# python run.py --config-name=cifar10

# =============================================
# Example 2: Single run with overrides
# =============================================
# python run.py --config-name=cifar10 \
#     training.lr=0.01 \
#     federated.num_clients=10 \
#     training.global_rounds=20

# =============================================
# Example 3: Multi-run sweep (all within this allocation)
# =============================================
# python run.py --config-name=cifar10 -m \
#     training.lr=0.0001,0.001,0.01 \
#     federated.alpha=0.1,0.5,1.0

# =============================================
# Example 4: Use predefined sweep config
# =============================================
# python run.py --config-name=sweep_cifar10 -m

# =============================================
# Example 5: CIFAR-100 experiment
# =============================================
# python run.py --config-name=cifar100

# =============================================
# Default: Run with CIFAR-10 config
# =============================================
# python run.py --config-name=cifar10 -m

# =============================================
# Example 6: Run multiple configs matching pattern (nc*cpr*.yaml)
# Uses a bash loop to run each config sequentially
# =============================================
# for config in nc5cpr3 nc20cpr5 nc100cpr10; do
#     echo "=========================================="
#     echo "Running config: $config"
#     echo "=========================================="
#     python run.py --config-path=../config/cifar10 --config-name=$config
# done

python run.py --config-path=/home/hli54/Projects/FCIL/config/cifar10 --config-name=nc100cpr10 -m


echo "Job completed at $(date)"
