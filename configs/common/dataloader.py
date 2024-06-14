from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from detectron2.config import LazyCall as L
from torch.utils.data.distributed import DistributedSampler
from data import GetDataTrain, DataGeneratorPromptFGSali


# Dataloader
train_dataset = DataGeneratorPromptFGSali(
    data=GetDataTrain(
        data_dir='/data4/yezixuan/Combined_Dataset',
        setting_dir='dataset_setting/comp1k_am2k_dis646_saliency_ref_him_triple_common',
    )
)

dataloader = OmegaConf.create()
dataloader.train = L(DataLoader)(
    dataset=train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    sampler=L(DistributedSampler)(
        dataset=train_dataset,
    ),
    drop_last=True
)
