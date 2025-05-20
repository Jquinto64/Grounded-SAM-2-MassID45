#!/bin/bash
#SBATCH -p rtx6000           # partition: should be gpu on MaRS, and a40, t4v1, t4v2, or rtx6000 on Vaughan (v)
#SBATCH --gres=gpu:1    # request GPU(s)
#SBATCH -c 4              # number of CPU cores
#SBATCH --mem=16G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --qos=m2
#SBATCH --time=8:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=grounded_sam2_zero_shot_sahi_inference_512_sahi_tiled_v9_025_conf_final_val_set_results_0.6_overlap_insect_alt_obj_fcn
## SBATCH --exclude=gpu177,gpu127

source ~/.bashrc
source activate gdino_testing
module load cuda-12.1

SEED="$SLURM_ARRAY_TASK_ID"

# Debugging outputs
pwd
which conda
python --version
pip freeze


# Run inference using GDINO + SAM 2.1
TILE_SIZE=512
SET="test" # val or test

python sahi_inference_gdino.py --gd_checkpoint gdino_checkpoints/groundingdino_swinb_cogcoor.pth \
--sam2_checkpoint checkpoints/sam2.1_hiera_large.pt \
--gd_config grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py \
--sam2_config configs/sam2.1/sam2.1_hiera_l.yaml \
--text_prompt "insect." \
--exp_name grounded_sam2_zero_shot_${TILE_SIZE}_tiled_0.6_overlap_insect_alt_obj_fcn_${SET}_SET \
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

# # For running with super-resolution
# SR_METHOD=real_esrgan
# python sahi_inference_gdino.py --gd_checkpoint gdino_checkpoints/groundingdino_swinb_cogcoor.pth \
# --sam2_checkpoint checkpoints/sam2.1_hiera_large.pt \
# --gd_config grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py \
# --sam2_config configs/sam2.1/sam2.1_hiera_l.yaml \
# --text_prompt "insect." \
# --exp_name grounded_sam2_zero_shot_2x_zoom_SR_${SR_METHOD}_0.6_overlap_insect_alt_obj_fcn \
# --dataset_json_path /h/jquinto/lifeplan_b_v9_cropped_center/annotations/instances_val2017.json \
# --dataset_img_path /scratch/ssd004/scratch/jquinto/lifeplan_b_v9_cropped_center_sr_${SR_METHOD}/val2017 \
# --crop_fac 16 \
# --postprocess_match_threshold 0.5 \
# --model_confidence_threshold 0.25 \
# --predict \
# --scale_factor 2 \
# --super_resolution \
# --slice_height ${TILE_SIZE} \
# --slice_width ${TILE_SIZE} \
# --overlap 0.6
