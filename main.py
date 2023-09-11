from train import *
from pred import *
from interpolation import *
from omegaconf import DictConfig, OmegaConf
from torchsummary import summary
from utilities import *

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Press 1 to train SR model")
    print("Press 2 to predict from SR model")
    print("Press 3 to calculate average PSNR in test set")

    user_input = input()
    if user_input == '1':
        task = Task.init(project_name='SR_FOR_CT_IMGs', 
                    task_name='SR_Train',
                    task_type=Task.TaskTypes.training,
                    reuse_last_task_id=False,
                    auto_resource_monitoring=False,
                    auto_connect_frameworks={"pytorch": False} # does not upload all the output models
                    
                    )

        dataset = BaseDataset(**cfg.dataset)
        Check_data.check_dataset(dataset)

    elif user_input == '2':
        inference(cfg)

    elif user_input == '3':
        calc_average_psnr_in_testset(cfg)
    #start_interpolate(cfg)

if __name__ == "__main__":
    main()