#!/bin/bash
#SBATCH -p rtx6000           # partition: should be gpu on MaRS, and a40, t4v1, t4v2, or rtx6000 on Vaughan (v)
#SBATCH --gres=gpu:1    # request GPU(s)
#SBATCH -c 8              # number of CPU cores
#SBATCH --mem=16G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --qos=normal
#SBATCH --time=16:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=florence2_sam2_zero_shot_sahi_inference_sahi_tiled_v9_025_conf_final_val_set_results_0.6_overlap_small_brown_yellow_insects_filter_0.6_alt_obj_fcn_no_duplicates_1024_resize_config
#SBATCH --exclude=gpu177

source ~/.bashrc
source activate grounded_sam2
module load cuda-12.1

SEED="$SLURM_ARRAY_TASK_ID"

# Debugging outputs
pwd
which conda
python --version
pip freeze

# Run inference using Florence-2 and SAM 2.1 
TILE_SIZE=512
SET="test" # 'test' or 'val'

python sahi_inference_florence2.py --florence2_model_id microsoft/Florence-2-large-ft \
--sam2_checkpoint checkpoints/sam2.1_hiera_large.pt \
--sam2_config configs/sam2.1/sam2.1_hiera_l.yaml \
--task_prompt "<OPEN_VOCABULARY_DETECTION>" \
--text_prompt "small brown yellow insects" \
--exp_name florence2_sam2_zero_shot_${TILE_SIZE}_tiled_0.6_overlap_small_brown_yellow_insects_filter_0.4_alt_obj_fcn_no_duplicates_${SET}_SET \
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