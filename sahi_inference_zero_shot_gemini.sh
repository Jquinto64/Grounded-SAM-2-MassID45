#!/bin/bash
#SBATCH -p rtx6000           # partition: should be gpu on MaRS, and a40, t4v1, t4v2, or rtx6000 on Vaughan (v)
#SBATCH --gres=gpu:1    # request GPU(s)
#SBATCH -c 8              # number of CPU cores
#SBATCH --mem=16G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=gemini_2_flash_sam2_zero_shot_sahi_inference_512_sahi_tiled_v9_025_conf_final_val_set_results_0.6_overlap_REPROD
#SBATCH --exclude=gpu177,gpu121,gpu127

source ~/.bashrc
source activate gdino_testing
module load cuda-12.1

SEED="$SLURM_ARRAY_TASK_ID"

# Debugging outputs
pwd
which conda
python --version
pip freeze

# Runs inference using Gemini Flash 2.0
TILE_SIZE=512
SET="val" # 'val' or 'test'
MODEL_NAME="gemini-2.0-flash" 

python sahi_inference_gemini.py --model_name ${MODEL_NAME} \
--sam2_checkpoint checkpoints/sam2.1_hiera_large.pt \
--sam2_config configs/sam2.1/sam2.1_hiera_l.yaml \
--text_prompt "Detect the 2d bounding boxes of the small brown insects, ants, flies, and/or gnats. Exclude loose wings, legs, and debris." \
--temperature 0.5 \
--exp_name ${MODEL_NAME}_sam2_zero_shot_${TILE_SIZE}_tiled_0.6_overlap_REPROD \
--dataset_json_path /h/jquinto/lifeplan_b_v9_cropped_center/annotations/instances_${SET}2017.json \
--dataset_img_path /h/jquinto/lifeplan_b_v9_cropped_center/${SET}2017 \
--crop_fac 16 \
--postprocess_match_threshold 0.5 \
--model_confidence_threshold 0.25 \
--predict \
--scale_factor 1 \
--slice_height ${TILE_SIZE} \
--slice_width ${TILE_SIZE} \
--overlap 0.6