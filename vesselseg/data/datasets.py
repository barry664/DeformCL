from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.file_io import PathManager
import os
import logging
logger = logging.getLogger(__name__)

def load_npz_cta_dataset(data_dir, size=None):
    dataset_dicts = []
    files = PathManager.ls(data_dir)
    suffix = ".npz"
    files = [f for f in files if f.endswith(suffix)]
    files = sorted(files, reverse=False)
    if size is not None:
        files = files[:size]
    for f in files:
        basename = f[: -len(suffix)]
        ret = dict(
            file_name=os.path.join(data_dir, f),
            file_id=basename
        )
        dataset_dicts.append(ret)

    logger.info(f"Get {len(dataset_dicts)} data from {data_dir}.")

    return dataset_dicts


DatasetCatalog.register(
    'HANS40',
    lambda data_dir='./HaN-Seg':
    load_npz_cta_dataset(data_dir)
)

MetadataCatalog.get('HANS40').set(
    data_dir='./HaN-Seg',
)

