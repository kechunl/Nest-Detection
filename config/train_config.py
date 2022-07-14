from detectron2 import model_zoo
import os
import math


def add_train_config(cfg, args):
    '''
    :param cfg: default config
    :param args: attr to be added into cfg: num_classes, loss, weight, instance, patch_dir, resize_dir, output_dir
    :return:
    '''
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("nest_train",)
    cfg.DATASETS.TEST = ("nest_train", "nest_val", "nest_test",)

    cfg.DATALOADER.NUM_WORKERS = 8
    # cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.INPUT.ResizeShortestEdge = True
    # -----------------------------------------------------------------------------
    # SOLVER
    # -----------------------------------------------------------------------------
    # patch1000_0.5_0.5_EP: 1133 Images left with annotation
    # patch1000_0.5_0.5_DE: 1280 Images left with annotation
    # patch1000_0.5_0.5: 1822 Images left with annotation, 4245 in total

    # corrected
    # patch1000_0.5_0.5: 2202 Images left with annotation in train, 380 in val, 602 in test
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.CHECKPOINT_PERIOD = 367
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = cfg.SOLVER.CHECKPOINT_PERIOD * 15
    cfg.SOLVER.STEPS = [cfg.SOLVER.CHECKPOINT_PERIOD * epoch for epoch in [4, 8, 12, 16, 20, 24, 28, 32, 36]]
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.WARMUP_ITERS = cfg.SOLVER.CHECKPOINT_PERIOD * 3

    cfg.TEST.EVAL_PERIOD = cfg.SOLVER.MAX_ITER

    # -----------------------------------------------------------------------------
    # MODEL
    # -----------------------------------------------------------------------------
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = '/projects/patho1/Kechun/NestDetection/result_corrected/ablations_10_times/ITER_3_WFOCAL_w_2_nest_5x_cat_2/model_0005504.pth'

    assert hasattr(args, 'num_classes')
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes  # foreground

    assert hasattr(args, 'loss')
    assert hasattr(args, 'weight')
    loss_type = args.loss

    # Loss for RPN BBOX CLASSIFICATION. options are ['CE', 'WCE', 'WFOCAL']
    cfg.MODEL.RPN.BBOX_CLS_LOSS_TYPE = loss_type
    cfg.MODEL.RPN.BBOX_CLS_LOSS_WEIGHT = args.weight  # weight for object

    # Loss for ROI BBOX CLASSIFICATION. options are ['CE', 'WCE', 'WFOCAL']
    cfg.MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_TYPE = loss_type
    cfg.MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_WEIGHT = [args.weight] * cfg.MODEL.ROI_HEADS.NUM_CLASSES + [1] # weight for class

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    cfg.FOCAL_LOSS_GAMMA = 2

    # -----------------------------------------------------------------------------
    # DIR
    # -----------------------------------------------------------------------------
    assert hasattr(args, "instance")
    # cfg.instance = "EP"
    cfg.instance = args.instance

    assert hasattr(args, "patch_dir")
    assert hasattr(args, "resize_dir")
    cfg.patch_dir = args.patch_dir
    cfg.resize_dir = args.resize_dir
    # cfg.patch_dir = '/projects/patho1/Kechun/NestDetection/dataset/patch/patch_1000_0.5_0.5_3'
    # cfg.resize_dir = '/projects/patho1/Kechun/NestDetection/dataset/full_image/ROI_cat_3_0.5'

    # if loss_type == 'CE':
    #     cfg.OUTPUT_DIR = '/projects/patho1/Kechun/NestDetection/result/' + loss_type + '_{}_5x_cat_{}_boxratio_{}'.format(cfg.instance, cfg.MODEL.ROI_HEADS.NUM_CLASSES+1, len(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[0]))
    # else:
    #     cfg.OUTPUT_DIR = '/projects/patho1/Kechun/NestDetection/result/' + loss_type + '_w_{}_{}_5x_cat_{}_boxratio_{}'.format(
    #         ''.join(map(str, cfg.MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_WEIGHT)), cfg.instance, cfg.MODEL.ROI_HEADS.NUM_CLASSES+1, len(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[0]))
    #
    # if cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS is False:
    #     cfg.OUTPUT_DIR += '_keepneg'
    # if cfg.INPUT.ResizeShortestEdge is True:
    #     cfg.OUTPUT_DIR += '_resize'
    # else:
    #     cfg.OUTPUT_DIR += '_noresize'

    assert hasattr(args, "output_dir")
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)