### 谣言检测

===========================
#### 00-项目信息
```
作者：TuringEmmy
时间:2021-04-18 22:52:28
详情：使用paddle的lstm对谣言进行检测,对项目进行了解耦操作，可是使用与任何拖拽式的交互软件使用
```
#### 01-环境依赖
```
ubuntu18.04
python3.7
paddle2.0
```
#### 02-部署步骤

```
sh scripts/data.sh  
sh scripts/predict.sh  
sh scripts/train.sh
```

#### 03-目录结构描述
```
.
├── data.py
├── model.py
├── predict.py
├── readme.md
├── run.py
├── scripts
│   ├── data.sh
│   ├── predict.sh
│   └── train.sh
└── utils.py

```


#### 04-版本更新
##### V1.0.0 版本内容更新
- 实现数据处理
- 模型训练
- 模型预测

#### 05-TUDO
- 封装api方便开发人员调用