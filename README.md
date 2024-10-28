# 2024-SJTU-AI-GoodsLine2Real
图转真挑战：AI绘制实体商品图 第三名(0.7628)思路与开源代码

使用的模型
- clip: 请从controlnet官方下载或自动下载
- sd1.5_canny: 预训练的时候使用的从controlnet[官方huggingface仓库](https://huggingface.co/lllyasviel/ControlNet/tree/main/models)下载或自动下载
- final：poolnet的推理模型，请自行在poolnet仓库下载（PoolNet-ResNet50 w/o edge model）
- 赛题训练的sd生成模型epoch=7-step=3999.zip：[百度网盘](https://pan.baidu.com/s/1NfSQCCDE7BLqzZ6UCgVnuQ?pwd=vlhi)（提取码：vlhi）

# 方法说明

**模型训练的方法学**：模型微调+提示词学习+两阶段推理

**模型微调**：使用controlnet对主办方提供的训练数据本身进行简单的微调训练

**提示词学习**：通过数据分析发现商品背景白色为主，为提示词添加统一的系统级提示引导

**两阶段推理**：使用poolnet对controlnet推理得到的商品生成图像进行后处理，通过显著性检测将背景强制设置为白色

## 数据分析

### 测试集注意事项

有一个数据24807.png没有，如果直接按range顺序直接生成图片会导致id不对应，提交错误

### 文本长度分析

字符长度

<img src="https://s2.loli.net/2024/10/28/sQ2letLf8W4wkBu.png" alt="image-20241028150725087" style="zoom:50%;" />

单词长度

<img src="https://s2.loli.net/2024/10/28/Hdcjxps369iQroa.png" alt="image-20241028150734488" style="zoom:50%;" />

文本内容分析

一共20183个词，小写化

```SQL
for: 10569
black: 1611
with: 1304
holder: 1176
home: 1101
diy: 1058
pieces: 1030
blue: 1000
white: 971
decor: 934
...
```

<img src="https://s2.loli.net/2024/10/28/ZWe1NAtm7S5BphI.png" alt="image-20241028150752178" style="zoom:50%;" />

对top200，单词出现206词以上的进行分类

虚词类 for with and For to

数字类 2 3 1 4

字母类 T L S M a b

分量类 Pieces Parts 2pcs 2x xl

对象类 Women Kids Women's Organizer Cat baby pet dog men womens girls animal children

身体类 Hair Head neck tee hand nail

穿着类 Shirt Jewelry Dress ring pendant hat sleeve chain clothes cap necklace beads makeup

形容类 DIY Decor Light Style Portable Decoration Gift Mini Round Durable electric small ornament waterproof adjustable long gifts accessory cute universal top large short fashion casual wear

材质类 Steel Metal Wooden Soft Silver Stainless led silicone wood plastic glass leather resin

颜色类 Black Blue White Red Green Pink gold gray brown orange

用品类 Holder Cover Bag Toy Set Tool Accessories Storage Bike Flower Case Clip Water Toys Hanging Stand Handle Replacement Ball Table Display Kit model craft strap making lamp bottle cable guitar doll coffee protector repair brush hook tray adapter pad board rack tools belt cup crafts seat air key mat chair motorcycle bar usb alloy boat statue pot bicycle costume supplies machine rod sewing container oil pin card

场景类 Home Car Party Outdoor Office Fishing Travel Summer Wall Kitchen Wedding Camping Room Christmas garden sports door golf game holiday birthday art hiking bedroom beach indoor training bathroom living work walking

### 图像尺寸分布

**长基本一致**

<img src="https://s2.loli.net/2024/10/28/Ymv3qyTe9dQSMZK.png" alt="image-20241028150800238" style="zoom:50%;" />

**长宽比基本一致**

<img src="https://s2.loli.net/2024/10/28/RkILvPymNFZJt9c.png" alt="image-20241028150810606" style="zoom:50%;" />

### 颜色

**背景平均颜色**，几乎全都是白色

r    0.977288    g    0.976632    b    0.975944

<img src="https://s2.loli.net/2024/10/28/FEUK7NCAuOY6iqv.png" alt="image-20241028150903143" style="zoom:50%;" />

**全局平均颜色**，几乎全部偏亮

r    0.737227   g    0.707953    b    0.693699

<img src="https://s2.loli.net/2024/10/28/oAKDnPiwdMjWst8.png" alt="image-20241028150817426" style="zoom:50%;" />

## 数据预处理

图像：全部resize到512x512大小保持大小一致

文本：添加系统级提示词，实现商品摄影和高亮色调引导

无其他处理

## 模型架构

(线条图像,修改后提示词) -> [controlnet] -> (初步生成图像) -> [poolnet] -> (最终生成图像) 

# 操作指南

## 环境依赖

环境基于controlnet官方代码 https://github.com/lllyasviel/ControlNet 所提供的环境进行安装

在参赛者提交的代码文件中，环境安装代码在environment.yaml中，使用如下方法安装

```SQL
conda env create -f environment.yaml
conda activate 2024_jdds_aigc
```

本模型第二阶段使用的poolnet参考了 https://github.com/backseason/PoolNet 不用安装其他依赖

## 执行代码

### **训练代码（controlnet）**

直接使用train.py

运行方法

```SQL
conda activate 2024_jdds_aigc
cd 项目文件夹
python train.py
```

train.py 重要代码简单解读

```Python
# 这是首次微调训练的原模型，使用contronnet的canny模型进行微调
# resume_path = './models/control_sd15_canny.pth'
# 这是已经微调训练后进行继续训练
resume_path = '/home/u1120230288/projects/2024-交大电商-AI绘制实体商品图/code/lightning_logs/version_3/checkpoints/epoch=7-step=3999.ckpt'
# 批量大小、日志输出频率、学习率、其他配置
batch_size = 20
logger_freq = 500
learning_rate = 1e-4
sd_locked = True
only_mid_control = False

# 设置数据集的地址
dataset = TrainDataset("数据集文件夹/Line_Drawing_to_Realistic_Drawing/train_data")
dataloader = DataLoader(dataset, num_workers=14, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
# 这里我使用了两场a800进行训练
trainer = pl.Trainer(strategy='ddp', gpus=2, precision=32, callbacks=[logger])
trainer.fit(model, dataloader)
```

### **推理代码（controlnet）**

使用inference.py进行推理即可，代码和train.py基本一致

运行方法

```Python
conda activate 2024_jdds_aigc
cd 项目文件夹
python inference.py
```

注意数据集地址修改为测试集

```Python
dataset = ValidDataset("数据集文件夹/data/Line_Drawing_to_Realistic_Drawing/valid_data")
```

### **推理代码（poolnet）**

由于上一步骤的生成图像保存在 项目文件夹/log_valid/日期-时间(mmdd-hhmm)/Realistic Drawing'文件夹中，复现时需要先在poolnet的main.py里面修改上一步生成的图像的保存位置

```Python
elif sal_mode == 'new':
    # 修改control生成图像的位置，例如
    image_root = '项目文件夹/log_valid/1015-2017/Realistic Drawing'
    image_source = None
```

然后，使用demo.sh直接进行推理即可

```Python
conda activate 2024_jdds_aigc
cd 项目文件夹/PoolNet
sh demo.sh
```

最终生成的图像保存在

`./PoolNet/results/run-new`当中，重命名Realistic Drawing然后压缩即可提交最终结果

# 性能评估

## 排行榜性能

微调+提示词+显著性检测获取生成图像掩码设置为白色背景=0.7628285

<img src="https://s2.loli.net/2024/10/28/PMeEqYg71OGNC4i.png" alt="image-20241028150931088" style="zoom: 67%;" />

## 计算平台

计算平台由南开大学高性能计算服务平台提供技术支持

## 算力使用

使用A800 80G*2进行训练

## 运算时间

### 训练时间

训练iteration约15000次,4000次约5.3小时，总计训练时间约15000/4000*5.3=20小时

<img src="https://s2.loli.net/2024/10/28/3vRLQAS7maPVent.png" alt="image-20241028150950779" style="zoom:50%;" />

### 推理时间

两张卡同时推理9999张图像，controlnet时间约三小时，poolnet时间约一小时

(3+1)\*60\*60/(9999/2)=2.88s

符合主办方要求的低于5s时间

## 部分生成效果展示

<img src="https://s2.loli.net/2024/10/28/kionwf92ZmMUEFP.png" alt="image-20241028150610769" style="zoom:50%;" />
