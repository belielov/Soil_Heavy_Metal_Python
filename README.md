# Soil_Heavy_Metal_Python
``` 
HMs/
├── dataset/                   # 数据存放目录
│   └── input.tif              # 输入的多光谱卫星影像 (11+ 波段)
│   └── output_prediction.tif  # 运行结束后生成的预测结果
├── src/                       # 源代码目录
│   └── main.py                # 执行推理的 Python 主程序
├── model/                     # 模型资源目录
│   ├── v5_xgb_model.json      # 训练好的 XGBoost 模型
│   └── scaler_params.json     # 标准化均值与缩放系数
└── README.md                  # 项目说明文档
```