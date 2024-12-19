#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

module purge
module load gcc/11.2.0
module load mambaforge
mamba activate coherence-tuning # create this if needed

cd /projects/$USER/porc/src/

# for loss in hinge ipo nca_pair robust
# do
	# python tune_model.py --output_dir "/scratch/alpine/${USER}/porc/gpt2" --loss_fn $loss
# done

python tune_model.py \
        --output_dir "/scratch/alpine/${USER}/porc/gpt2" \
        --dataset "lecslab/porc-gpt2-v1" \
        --loss_fn robust \
        --label_smoothing 0.048
