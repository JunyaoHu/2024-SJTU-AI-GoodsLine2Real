# 2024-SJTU-AI-GoodsLine2Real
图转真挑战：AI绘制实体商品图 第三名(0.7628)思路与开源代码

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

<img src="https://yvddwzr9hs.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDZiOGYyYmRhMGM5Y2QxYjAxNmI0NzBhZGQ0YWVmNTZfS0VZbHJxaU5UeU5xRW5yWkxTS3EyRUhNWTFMemQ1bTJfVG9rZW46S3lFNmJIcHF4b2drNFd4dzZLQWNnRUxMbnpnXzE3MzAwOTkwMzQ6MTczMDEwMjYzNF9WNA" alt="img" style="zoom:50%;" />

单词长度

<img src="https://yvddwzr9hs.feishu.cn/space/api/box/stream/download/asynccode/?code=NGExNGZhMzE4YzVmYmE1Njc2NTQ3Yzk0OTcyZGU4ODVfcktHZ0pucGI2RE1EZFFPeEJMakU3Wmd5SE93RDNqOHlfVG9rZW46WDVJVGJJQ2xVb3VoUUZ4MGZRY2M5ZmlQbnZiXzE3MzAwOTkwMzQ6MTczMDEwMjYzNF9WNA" alt="img" style="zoom:50%;" />

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

<img src="https://yvddwzr9hs.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDIyYTM4OTJhMTIzN2Y3MjZkMjdkNWQ2NGJkMDgzMTZfYWdJSWYxQ09zaGt5anVWR29pQmdOY1FkejlST0o1bWZfVG9rZW46TWpLYmJ1RDlTb2pvZHd4Nk1IMGNweTdPbmxkXzE3MzAwOTkwMzQ6MTczMDEwMjYzNF9WNA" alt="img" style="zoom:50%;" />

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

<img src="https://yvddwzr9hs.feishu.cn/space/api/box/stream/download/asynccode/?code=OGQxOTdiNmM1Mzk1OThlZDIyZmFhZGZlOGRkNTE5NjRfbjlJalc5SUpKTGV1YkszYTVaV2FQU3dsUHdQYzE5WEdfVG9rZW46T3RoWGI3REFqbzB5eFB4clZhRGNmWXFTblRlXzE3MzAwOTkwMzQ6MTczMDEwMjYzNF9WNA" alt="img" style="zoom:50%;" />

**长宽比基本一致**

![img](https://yvddwzr9hs.feishu.cn/space/api/box/stream/download/asynccode/?code=MzhkNzViMTM2YzZmYWNlZjYxM2ZkODRlMTc5OGJlMjRfVThnZHlKU3lpVXM3bmpuRjBiT3VjeVhpT3hWQlZjbnFfVG9rZW46Vno2T2JHeHlqb1RyTW54WmxNRWNSclE2bmFlXzE3MzAwOTkwMzQ6MTczMDEwMjYzNF9WNA)

### 颜色

**背景平均颜色**，几乎全都是白色

r    0.977288    g    0.976632    b    0.975944

<img src="https://yvddwzr9hs.feishu.cn/space/api/box/stream/download/asynccode/?code=NjVlNWY3NjUwYjM5YTBhMmQ0MGY2ZTQ2ZmIzYTM3YTFfYU5XYThScnBXb1lIdkVEdGs2NlloYXJtcWpGR01nSUxfVG9rZW46S2ROZWJZQ0Rob3RWZVF4QWVPNGNRTEFhbjRlXzE3MzAwOTkwMzQ6MTczMDEwMjYzNF9WNA" alt="img" style="zoom:50%;" />

**全局平均颜色**，几乎全部偏亮

r    0.737227   g    0.707953    b    0.693699

<img src="https://yvddwzr9hs.feishu.cn/space/api/box/stream/download/asynccode/?code=YjczZTE5ZTMyNDQ1NGQ1NDg1OTQ1MjY3NzZjYmVjOGZfQ2dIYWYwdHFpQ0RSS25Yd0JSbFA0UmxRbFVpTkkwNUVfVG9rZW46SnF6YmIxQ3N4bzZWMHV4cjh0amNORTRybkJnXzE3MzAwOTkwMzQ6MTczMDEwMjYzNF9WNA" alt="img" style="zoom:50%;" />

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

<img src="https://yvddwzr9hs.feishu.cn/space/api/box/stream/download/asynccode/?code=YzZjNTAwODA2ODQwM2NjZGQ4MGZhMWRjYzg5NTFjNTdfWGc4SDF3RWRjUGNCYVNLRWhJeGdEUDFqVVhxODFNam1fVG9rZW46V29nZGJhY1NpbzJhM3Z4QmwxQ2N5a3Vnbk5iXzE3MzAwOTkwMzQ6MTczMDEwMjYzNF9WNA" alt="img" style="zoom:50%;" />

## 计算平台

计算平台由南开大学高性能计算服务平台提供技术支持

## 算力使用

使用A800 80G*2进行训练

## 运算时间

### 训练时间

训练iteration约15000次

<img src="https://yvddwzr9hs.feishu.cn/space/api/box/stream/download/asynccode/?code=NTQ2NzE1NTE1MmVjODNjOTFlOWZhYmU4MTZjZjZiYzBfZDI2QlpDS2Y2Tk9pWjE2SjFxd3FGMUtXb0YydXYybHpfVG9rZW46SGpSSWJUTlhHb1NNcHR4aEZJeWNjTEpIbkRmXzE3MzAwOTkwMzQ6MTczMDEwMjYzNF9WNA" alt="img" style="zoom:50%;" />

4000次约5.3小时，总计训练时间约15000/4000*5.3=20小时

![img](https://yvddwzr9hs.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDFkZDVmYTg2ODA2MWU0MDgxMTVmNzNkYmI3NzRiZGFfblEzem9zVktQRUVkMDM2RzNIY2k3aG5CNEZ4b1VoWmVfVG9rZW46UklLZmJFT21GbzdUckd4RW1wWmNTUGV5bkxlXzE3MzAwOTkwMzQ6MTczMDEwMjYzNF9WNA)

### 推理时间

两张卡同时推理9999张图像，controlnet时间约三小时，poolnet时间约一小时

(3+1)*60*60/(9999/2)=2.88s

符合主办方要求的低于5s时间

## 部分生成效果展示

<img src="https://yvddwzr9hs.feishu.cn/space/api/box/stream/download/asynccode/?code=YTJlNjc4NTQyNWE5MTlkNmQzZmRmMGRkOWQxZjljZDJfQVE1M1dWY0ZreGxtUE5XdFJlSHcyWW5MV0xrdFdtSXZfVG9rZW46V2hNdWIyZ09HbzE4b3B4TkVzWGN1eHVZbktjXzE3MzAwOTkwMzQ6MTczMDEwMjYzNF9WNA" alt="img" style="zoom:50%;" />
