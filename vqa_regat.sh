#!/bin/bash
#SBATCH -A kanishk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3000
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --job-name=vqa_regat
#SBATCH --mail-user=kanishk.jain@alumni.iiit.ac.in
#SBATCH --mail-type=ALL

module load python/3.7.4
module load cuda/10.0
module load cudnn/7-cuda-10.0

set -e

if [ -d "/ssd_scratch/cvit/kanishk/vqa" ]; then
    echo "folder exists, proceed with training"
else
    mkdir -p /ssd_scratch/cvit/kanishk/
    rm -rf /ssd_scratch/cvit/kanishk/*

    echo "copying features from share3 to ssd_scratch"
    scp -r kanishk@ada:/share3/kanishk/vqa /ssd_scratch/cvit/kanishk/
    echo "copied features from scratch3"
fi

#python3 main.py --config config/butd_vqa.json
python3 eval.py --output_folder pretrained_models/regat_implicit/ban_1_implicit_vqa_196
