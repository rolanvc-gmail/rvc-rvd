import torch
import os
import config
from data import load_data
from models.unet import Unet


def get_dim(data_config_):
    return 48 if data_config_["img_size"] == 64 else 64

def get_main_mults(data_config_):
    return (1,1,2,2,4,4) if data_config_["img_size"] in [128, 256] else (1,2,4,8)

for data_config in config.data_configs:
    train_data, val_data = load_data(
        data_config, config.BATCH_SIZE, pin_memory=False, num_workers=8, distributed=False
    )

    for pred_mode in config.pred_modes:
        for transform_mode in config.transform_modes:
            model_name = f"{config.backbone}-{config.optimizer}-{pred_mode}-l1-{data_config['dataset_name']}-d{get_dim(data_config)}-t{config.iteration_step}-{transform_mode}-al{config.aux_loss}{config.additional_note}"
            results_folder = os.path.join(config.result_root, f"{model_name}")
            loaded_param = torch.load(
                str(f"{results_folder}/{model_name}_{0}.pt"),
                map_location=lambda storage, loc: storage
            )

            denoise_model = Unet(
               dim=get_dim(data_config),
               context_dim_factor=config.context_dim_factor,
               channels=data_config["img_channel"],
               dim_mults=get_main_mults(data_config),
           )
