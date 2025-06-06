import logging
from collections import OrderedDict

import detectron2.utils.comm as comm
from train_utils import build_adamw_optimizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import DatasetEvaluators, DatasetEvaluator, print_csv_format
from detectron2.utils.logger import setup_logger

from vesselseg.config import add_seg3d_config
from vesselseg.evaluation import (
    CommonDiceEvaluator,
)
from vesselseg.data import (
    ClineDeformDatasetMapper,
    VesselSegDatasetMapper,
    build_cline_deform_transform_gen,
    build_bbox_transform_gen,
)
from train_utils import (
    build_train_loader,
    build_test_loader
)
from train_utils import (
    inference_on_dataset
)


def get_dataset_mapper(cfg, is_train=False):
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    if meta_arch != 'Bbox3d':
        mapper_func = ClineDeformDatasetMapper
        transform_builder = build_cline_deform_transform_gen
    else:
        mapper_func = VesselSegDatasetMapper
        transform_builder = build_bbox_transform_gen

    return mapper_func(cfg, transform_builder, is_train)


class Trainer(DefaultTrainer):

    def __init__(self, cfg):
        super(Trainer, self).__init__(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        evaluators = []
        evaluators.append(CommonDiceEvaluator(dataset_name, cfg))
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_test_loader(
            cfg, dataset_name, mapper=get_dataset_mapper(cfg, is_train=False)
        )

    @classmethod
    def build_train_loader(cls, cfg):
        return build_train_loader(
            cfg, mapper=get_dataset_mapper(cfg, is_train=True)
        )

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_adamw_optimizer(cfg, model)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator, amp=False)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_seg3d_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(),
                 name="VesselSeg3D", abbrev_name="vessel")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
