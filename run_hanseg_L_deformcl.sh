python train_net.py --num-gpus 4 --dist-url auto \
--config-file configs/deformcl.yaml \
MODEL.PRED_CLASS 1 \
INPUT.CROP_SIZE_TRAIN "(224, 144, 144)" \
SOLVER.MAX_ITER 6000 \
OUTPUT_DIR "outputs/DeformCL_L"