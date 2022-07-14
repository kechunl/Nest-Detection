import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_setup, default_argument_parser, launch, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format, NestEvaluator, inference_context
from detectron2.modeling import build_model
import detectron2.utils.comm as comm
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import CommonMetricPrinter, EventStorage, JSONWriter, TensorboardXWriter
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode

from data.prepare_coco import get_nest_dicts, get_raw_nest_dicts
from config.train_config import add_train_config

from detectron2.evaluation.nest_evaluation import *

import os, logging, cv2
import pandas as pd
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
logger = logging.getLogger("detectron2")


class Trainer(DefaultTrainer):
    @classmethod
    def get_evaluator(cls, cfg, dataset_name, output_folder=None, tiling=False, vis=False):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # evaluator = COCOEvaluator(dataset_name, ("bbox", "segm"), output_dir=output_folder, use_fast_impl=False, tiling=tiling)
        evaluator = NestEvaluator(dataset_name, num_classes=cfg.MODEL.ROI_HEADS.NUM_CLASSES, output_dir=output_folder, tiling=tiling, vis=vis)
        return evaluator

    @classmethod
    def do_tiling_test(cls, cfg, model, iteration):
        results = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:
            data_loader = build_detection_test_loader(cfg, dataset_name)    # batch size = 1
            evaluator = Trainer.get_evaluator(cfg, "raw_" + dataset_name, os.path.join(cfg.OUTPUT_DIR, "Image_inference_iter_{}".format(iteration), dataset_name), tiling=True, vis=True)
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
        # if len(results) == 1:
        #     results = list(results.values())[0]
        return results

    @classmethod
    def do_test(cls, cfg, model):
        results = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:
            data_loader = build_detection_test_loader(cfg, dataset_name)    # batch size = 1
            evaluator = Trainer.get_evaluator(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name), tiling=False, vis=False)
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
        return results

    @classmethod
    def vis_patch(cls, cfg, patch_path=None):
        predictor = DefaultPredictor(cfg)
        if patch_path is None:
            # vis datset
            for dataset_name in cfg.DATASETS.TEST:
                out_dir = os.path.join(cfg.OUTPUT_DIR, 'vis_{}_patch_{}'.format(os.path.basename(cfg.MODEL.WEIGHTS).split('.')[0], dataset_name))
                os.makedirs(out_dir, exist_ok=True)
                if dataset_name == "nest_test":
                    dset = "test"
                elif dataset_name == "nest_val":
                    dset = "val"
                elif dataset_name == "nest_fully_marked_test":
                    dset = "fully_marked_test"
                dataset_dicts = get_nest_dicts(dset)
                for idx, d in enumerate(dataset_dicts):
                    im = cv2.imread(d["file_name"])
                    outputs = predictor(im)
                    v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get(dataset_name),
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW)
                    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    cv2.imwrite(os.path.join(out_dir, os.path.basename(d["file_name"])), out.get_image()[:, :, ::-1])
        else:
            im = cv2.imread(patch_path)
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=MetadataCatalog.get("nest_test"),
                           scale=0.5,
                           instance_mode=ColorMode.IMAGE_BW)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, "mask_" + os.path.basename(patch_path)), out.get_image()[:, :, ::-1])

    @classmethod
    def do_train(cls, cfg, model, resume=False):
        model.train()
        optimizer = build_optimizer(cfg, model)
        scheduler = build_lr_scheduler(cfg, optimizer)

        checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
        start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
        max_iter = cfg.SOLVER.MAX_ITER

        periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

        writers = ([CommonMetricPrinter(max_iter), JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
                    TensorboardXWriter(cfg.OUTPUT_DIR), ]if comm.is_main_process() else [])

        data_loader = build_detection_train_loader(cfg)
        logger.info("Starting training from iteration {}".format(start_iter))
        with EventStorage(start_iter) as storage:
            for data, iteration in zip(data_loader, range(start_iter, max_iter)):
                storage.iter = iteration

                loss_dict = model(data)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()

                if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0 and iteration != max_iter - 1:
                    # results = Trainer.do_test(cfg, model)
                    # comm.synchronize()
                    # if comm.is_main_process():
                    #     put_scalars_types(storage, results)
                    tiling_results = Trainer.do_tiling_test(cfg, model, iteration)
                    comm.synchronize()
                    if comm.is_main_process():
                        put_scalars_types(storage, tiling_results, prefix='ImageLevel_')

                if iteration - start_iter > 5 and ((iteration + 1) % 100 == 0 or iteration == max_iter - 1):
                    for writer in writers:
                        writer.write()
                periodic_checkpointer.step(iteration)


def put_scalars_types(storage, results, prefix=''):
    for key in results.keys():
        result = results[key]
        for types in list(result.keys()):
            for m in list(result[types].keys()):
                storage.put_scalar("{}_{}_metrics/{}".format(prefix+key, types, m), result[types][m], smoothing_hint=False)


def setup(args):
    cfg = get_cfg()
    add_train_config(cfg, args)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    if cfg.MODEL.ROI_HEADS.NUM_CLASSES == 1:
        thing_classes = [cfg.instance]
    elif cfg.MODEL.ROI_HEADS.NUM_CLASSES == 2:
        thing_classes = ["EP_nest", "DE_nest"]
    for d in ["train", "val", "test"]:
        DatasetCatalog.register("nest_" + d, lambda d=d: get_nest_dicts(cfg.patch_dir, d))
        MetadataCatalog.get("nest_" + d).set(thing_classes=thing_classes)
        DatasetCatalog.register("raw_nest_" + d, lambda d=d: get_raw_nest_dicts(cfg.resize_dir, d))
        MetadataCatalog.get("raw_nest_" + d).set(thing_classes=thing_classes)

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume)
        return Trainer.do_tiling_test(cfg, model, 0)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(model, device_ids=[comm.get_local_rank()], broadcast_buffers=False)
    Trainer.do_train(cfg, model, resume=args.resume)
    result = Trainer.do_tiling_test(cfg, model, cfg.SOLVER.MAX_ITER)
    df = pd.DataFrame(result['nest_test'])
    df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'Test_result.csv'))
    return


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.num_gpus = 2
    args.resume = False
    args.eval_only = False

    # args.num_classes = 1
    # args.loss = "WFOCAL"
    # args.weight = 2
    # args.instance = "nest"
    # args.patch_dir = "/projects/patho1/Kechun/NestDetection/dataset/patch/patch_1000_0.5_0.5_{}".format(args.num_classes+1)
    # args.resize_dir = "/projects/patho1/Kechun/NestDetection/dataset/full_image/ROI_cat_{}_0.5".format(args.num_classes+1)
    # args.output_dir = "/projects/patho1/Kechun/NestDetection/result/ablations_old_split/nonopretrained_{}_w_{}_{}_5x_cat_{}".format(args.loss, args.weight, args.instance, args.num_classes+1)

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
