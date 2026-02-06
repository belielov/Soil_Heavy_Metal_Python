import os
import json
import numpy as np
import xgboost as bst
from osgeo import gdal

# 消除 GDAL 4.0 异常警告
gdal.UseExceptions()

def calculate_indices(bands):
    """
    光谱指数计算逻辑，输入 bands 为 (11, N) 的矩阵
    """
    # 转换为 0-1 范围的反射率 (对应 C++ 中的 / 10000.0f)
    b = bands / 10000.0
    eps = 1e-10
    
    # 计算特征指数
    ndvi = (b[10] - b[3]) / (b[10] + b[3] + eps)
    ndwi = (b[2] - b[10]) / (b[2] + b[10] + eps)
    
    L = 0.5
    savi = (1.0 + L) * (b[10] - b[3]) / (b[10] + b[3] + L + eps)
    ci = (b[3] - b[2]) / (b[3] + b[2] + eps)
    
    b11_b12_ratio = b[8] / (b[9] + eps)
    b05_b06_ratio = b[4] / (b[5] + eps)
    b11_b12_diff = b[8] - b[9]
    
    # 堆叠原始波段和新指数
    features = np.vstack([b, ndvi, ndwi, savi, ci, b11_b12_ratio, b05_b06_ratio, b11_b12_diff])
    return features.T # 返回 (N, 18)

def main():
    # 路径配置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "model", "v5_xgb_model.json")
    scaler_path = os.path.join(base_dir, "model", "scaler_params.json")
    input_path = os.path.join(base_dir, "dataset", "input.tif")
    # 输出至dataset目录
    output_path = os.path.join(base_dir, "dataset", "output_prediction.tif")

    # 1. 加载模型和标准化参数
    model = bst.Booster()
    model.load_model(model_path)
    # 设置多线程推理
    model.set_param('nthread', 0)
    
    with open(scaler_path, 'r') as f:
        scaler = json.load(f)
    mean = np.array(scaler['mean'])
    scale = np.array(scaler['scale'])

    # 2. 打开影像
    ds = gdal.Open(input_path)
    if ds is None:
        print(f"Error: Could not open {input_path}")
        return

    x_size = ds.RasterXSize
    y_size = ds.RasterYSize
    actual_band_count = ds.RasterCount
    projection = ds.GetProjection()
    geotransform = ds.GetGeoTransform()

    print(f"Total bands detected: {actual_band_count}") #

    # 3. 创建输出文件
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path, x_size, y_size, 1, gdal.GDT_Float32)
    out_ds.SetProjection(projection)
    out_ds.SetGeoTransform(geotransform)
    out_band = out_ds.GetRasterBand(1)

    # 4. 多行分块处理
    block_rows = 300
    print("Starting prediction...")

    for i in range(0, y_size, block_rows):
        rows = min(block_rows, y_size - i)
        
        # 读取数据
        block_data = ds.ReadAsArray(0, i, x_size, rows) 
        
        # 核心修复：截取前 11 个波段，解决 reshape 报错
        if actual_band_count > 11:
            block_data = block_data[:11, :, :]
        elif actual_band_count < 11:
            raise ValueError(f"Error: input.tif only has {actual_band_count} bands, 11 required.")

        pixels_count = rows * x_size
        flat_bands = block_data.reshape(11, pixels_count)
        
        # 特征工程与标准化
        features = calculate_indices(flat_bands) 
        features = (features - mean) / scale
        
        # XGBoost 批量推理
        dmatrix = bst.DMatrix(features)
        preds = model.predict(dmatrix)
        
        # 指数还原并写入
        preds_exp = np.exp(preds).reshape(rows, x_size)
        out_band.WriteArray(preds_exp, 0, i)
        
        # 进度显示：每 10% 显示一次
        current_progress = int((i + rows) * 100 / y_size)
        # 逻辑：当前块跨越了 10% 的阈值时打印
        if (i == 0) or (current_progress // 10 > (i * 100 // y_size) // 10):
            print(f"Progress: {current_progress // 10 * 10}%")

    # 5. 清理
    out_band.FlushCache()
    ds = None
    out_ds = None
    print(f"\n[Success] Processing completed! Result saved to: {output_path}")

if __name__ == "__main__":
    main()