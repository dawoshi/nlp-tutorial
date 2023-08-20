#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: msra.sh

REPO_PATH="/data0/guoxinian/codes/mrc-for-flat-nested-ner-master"
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_DIR=$REPO_PATH/mrc_data/zh_msra
BERT_DIR=$REPO_PATH/pre_trained_model/bert-base-chinese
SPAN_WEIGHT=0.1
DROPOUT=0.2
LR=8e-6
MAXLEN=128
INTER_HIDDEN=1536
BATCH_SIZE=8
PREC=16
VAL_CKPT=0.25
ACC_GRAD=1
MAX_EPOCH=20
SPAN_CANDI=pred_and_gold
PROGRESS_BAR=1
BEST_CHECKPOINT=$REPO_PATH/output/zh_msra_bert_lr8e-620200913_dropout0.2_maxlen128/epoch=8_v0.ckpt
OUTPUT_DIR=$REPO_PATH/output/zh_msra_bert_lr
CUDA_VISIBLE_DEVICES=0 python $REPO_PATH/convert_onnx/mrc_ner_trainer.py \
--gpus="1" \
--distributed_backend=ddp \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAXLEN} \
--batch_size ${BATCH_SIZE} \
--precision=${PREC} \
--progress_bar_refresh_rate ${PROGRESS_BAR} \
--lr ${LR} \
--val_check_interval ${VAL_CKPT} \
--accumulate_grad_batches ${ACC_GRAD} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--weight_span ${SPAN_WEIGHT} \
--span_loss_candidates ${SPAN_CANDI} \
--chinese \
--workers 48 \
--classifier_intermediate_hidden_size ${INTER_HIDDEN} \
--best_checkpoint ${BEST_CHECKPOINT}
