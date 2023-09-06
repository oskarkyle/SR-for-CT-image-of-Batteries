from train import *
from pred import *
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Press 1 to train SR model")
    print("Press 2 to predict from SR model")

    user_input = input()
    if user_input == '1':
        task = Task.init(project_name='SR_FOR_CT_IMGs', 
                    task_name='SR_Train',
                    task_type=Task.TaskTypes.training,
                    reuse_last_task_id=False,
                    auto_resource_monitoring=False,
                    auto_connect_frameworks={"pytorch": False} # does not upload all the output models
                    
                    )

        start_training(cfg)
        
    elif user_input == '2':
        inference(cfg)


if __name__ == "__main__":
    main()