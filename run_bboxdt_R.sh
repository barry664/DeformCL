python train_net.py --num-gpus 4 --dist-url auto --resume \
--config-file configs/bbox.yaml \
MODEL.PRED_CLASS 2 \
INPUT.CROP_SIZE_TRAIN "(144, 288, 288)" \
SOLVER.MAX_ITER 6000 \
OUTPUT_DIR "outputs/BBOXDT_R"