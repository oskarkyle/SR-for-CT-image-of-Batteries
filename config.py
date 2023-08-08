import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs")
def train_config(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    


if __name__ == "__main__":
    train_config()