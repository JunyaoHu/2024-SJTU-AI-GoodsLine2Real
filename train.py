from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import TrainDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Configs
# resume_path = './models/control_sd15_canny.pth'
resume_path = '/home/u1120230288/projects/2024-交大电商-AI绘制实体商品图/code/lightning_logs/version_3/checkpoints/epoch=7-step=3999.ckpt'
batch_size = 20
logger_freq = 500
learning_rate = 1e-4
sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = TrainDataset("/home/u1120230288/projects/data/Line_Drawing_to_Realistic_Drawing/train_data")
dataloader = DataLoader(dataset, num_workers=14, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(strategy='ddp', gpus=2, precision=32, callbacks=[logger])
trainer.fit(model, dataloader)

"""
conda activate 2024_jdds_aigc
cd /home/u1120230288/projects/2024-交大电商-AI绘制实体商品图/code
python train.py

nvitop
"""