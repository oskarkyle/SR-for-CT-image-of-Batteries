from source.BaseDataset import BaseDataset
from source.data_utils import *
from omegaconf import DictConfig, OmegaConf
import hydra
import os
"""
hydra.main(version_base=None, config_path="configs", config_name="dataset")
def main(cfg: DictConfig)-> None:
    DS = BaseDataset(**cfg.dataset)
    if cfg.checkdataset:
        prepare_data.check_dataset(DS)
"""

if __name__ == "__main__":
    
    data_root = "/Users/haoruilong/BA_code/SR_for_CT_image_of_Batteries"#r'H:\SR_for_CT_image_of_Batteries'
    dataset_dir = "/dataset/pristine"#[r'\dataset\pristine']

    cfgs_path_p = data_root + '/configs/preprocess.yaml'
    cfgs_path_t = data_root + '/configs/transform.yaml'

    if os.path.exists(cfgs_path_p):
        preprocess_cfgs = OmegaConf.load(cfgs_path_p)
    else:
        preprocess_cfgs = None

    if os.path.exists(cfgs_path_t):
        transform_cfgs = OmegaConf.load(cfgs_path_t)
    else:
        transform_cfgs = None

    filelist = BaseDataset.load_file_paths_from_dir(data_root, dataset_dir)
    print(filelist)

    mydataset = BaseDataset('SR', 'train', 512, 4, dataset_dir, data_root, transform_cfgs, preprocess_cfgs)
    prepare_data.check_dataset(mydataset)
        