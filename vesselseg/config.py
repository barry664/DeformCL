from detectron2.config import CfgNode as CN


def add_seg3d_config(cfg):
    # We use n_fold cross validation for evaluation.
    # Number of folds
    cfg.DATASETS.NUM_FOLDS = 5
    # IDs of test folds, every entry in [0, num_folds - 1]
    cfg.DATASETS.TEST_FOLDS = (0,)

    # categories
    cfg.MODEL.PRED_CLASS = 0
    cfg.MODEL.N_CONTROL_POINTS = 4
    cfg.MODEL.TASK = ['cline']

    cfg.MODEL.OUT_CHANNELS = (1, 1, 1, 1)  # background included in 11
    cfg.MODEL.OUT_TASKS = ["cline"]

    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.LOSS = 'dice'

    cfg.MODEL.UNETENCODER = CN()
    cfg.MODEL.UNETENCODER.BASE_CHANNELS = 16
    cfg.MODEL.UNETENCODER.NUM_LAYERS = 4
    cfg.MODEL.UNETENCODER.NORM = 'SyncBN'

    # config for segmentation
    cfg.MODEL.SEGMENTOR = CN()
    cfg.MODEL.SEGMENTOR.FOCAL_LOSS_GAMMA = 2.0
    cfg.MODEL.SEGMENTOR.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.SEGMENTOR.HEAD = False
    cfg.MODEL.SEGMENTOR.AORTA = False
    cfg.MODEL.SEGMENTOR.DIST_INPUT = False
    cfg.MODEL.SEGMENTOR.LOSS = "Diceloss"
    
    # deform
    cfg.MODEL.DEFORM = CN()
    cfg.MODEL.DEFORM.NUM_STEPS = 4
    cfg.MODEL.DEFORM.NORM = 'SyncBN'
    cfg.MODEL.DEFORM.LOSS_EDGE_WEIGHT = 1
    cfg.MODEL.DEFORM.SDF_LOSS_WEIGHT = 1
    cfg.MODEL.DEFORM.CHAMFER_LOSS_WEIGHT = 1
    cfg.MODEL.DEFORM.ADPTPL = True
    cfg.MODEL.DEFORM.LOWER_THRES = 30
    cfg.MODEL.DEFORM.PTS_NUM = 350
    cfg.MODEL.DEFORM.USE_LOCAL_CHAMFER = True

    # config for input
    cfg.INPUT.CROP_SIZE_TRAIN = (128, 128, 128)

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.CHECKPOINT_PERIOD = 2500
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.TEST.EVAL_PERIOD = 2500

    # non benchmark version of 3D convolution is very slow, so
    # we use CUDNN_BENCHMARK = True during training. Remember
    # to use fix input size to accelerate training speed.
    cfg.CUDNN_BENCHMARK = True
