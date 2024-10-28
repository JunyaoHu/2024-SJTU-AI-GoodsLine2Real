from share import *

import pytorch_lightning as pl 
from torch.utils.data import DataLoader
from dataset import TrainDataset, ValidDataset
from cldm.model import create_model, load_state_dict

# Configs
resume_path = '/home/u1120230288/projects/2024-交大电商-AI绘制实体商品图/code/lightning_logs/epoch=24-step=12499.ckpt'
batch_size = 16
sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
# dataset = TrainDataset("/home/u1120230288/projects/data/Line_Drawing_to_Realistic_Drawing/train_data")
dataset = ValidDataset("/home/u1120230288/projects/data/Line_Drawing_to_Realistic_Drawing/valid_data")
dataloader = DataLoader(dataset, num_workers=7, batch_size=batch_size, shuffle=False)

trainer = pl.Trainer(strategy='ddp', gpus=2, precision=32)
trainer.test(model=model, dataloaders=dataloader)

"""
conda activate 2024_jdds_aigc
cd /home/u1120230288/projects/2024-交大电商-AI绘制实体商品图/code
python inference.py

# 修改cldm
nvitop
"""