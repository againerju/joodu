#!/bin/sh

PROJ_ROOT=${PWD}

CUDA_VISIBLE_DEVICES=0 python eval.py \
--traj_pred_model_path ${PROJ_ROOT}/experiments/traj_pred/000_enc_dec/model/enc_dec_iros_2023.ckpt \
--ood_detection_model_path ${PROJ_ROOT}/experiments/ood_detection/000_lgmm/model/lgmm_iros_2023.joblib \
--uncertainty_model_path ${PROJ_ROOT}/experiments/uncertainty_estimation/000_e_reg/model/e_reg_iros_2023.ckpt \
--split eval \
