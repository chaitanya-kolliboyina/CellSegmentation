import torch
from dataclasses import dataclass

# GPU related parameters
@dataclass
class gpuConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus: int = torch.cuda.device_count()
    device_name :str = torch.cuda.get_device_name()
    device_properties :str = torch.cuda.get_device_properties()

#training parameters
@dataclass
class trainConfig:
    data_path: str = "./data/brain_tumor"
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    batch_size: int = 32
    epochs: int = 5
    lr: float = 2e-5
    num_classes: int = 2 #background & Cell 

# data paramets
@dataclass
class DataPaths:
    images_root: str
    masks_root: str

    