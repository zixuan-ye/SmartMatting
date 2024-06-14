from .common.train import train
# from .common.dataloader import dataloader
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .common.model import model



embed_dim, num_heads = 384, 6


lr_multiplier.scheduler.values=[1.0, 0.1, 0.05]
lr_multiplier.scheduler.milestones=[int(18747 / 16 / 2 * 30), int(18747 / 16 / 2 * 60)]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 500 / train.max_iter
train.ddp.find_unused_parameters=True


train.max_iter = int(18747 / 16 / 2 * 100)
train.checkpointer.period = int(18747 / 16 / 2 * 4)


train.output_dir = './output_of_train/exp_name'



train.eval_period = 2000