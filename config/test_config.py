from detectron2 import model_zoo
import os, math


def add_test_config(cfg, args):
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = '/projects/patho1/Kechun/NestDetection/result_corrected/ablations_10_times/ITER_3_WFOCAL_w_2_nest_5x_cat_2/model_final.pth'

    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.INPUT.ResizeShortestEdge = True

    cfg.DATASETS.TEST = ("test", "valid", "train")

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
    # CUSTOM parameters
    # -----------------------------------------------------------------------------
    assert hasattr(args, "instance")
    cfg.instance = args.instance

    assert hasattr(args, "patch_dir")
    assert hasattr(args, "output_dir")
    cfg.OUTPUT_DIR = args.output_dir
    cfg.PATCH_DIR = args.patch_dir
    assert hasattr(args, "resize_dir")
    cfg.RESIZE_DIR = args.resize_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)