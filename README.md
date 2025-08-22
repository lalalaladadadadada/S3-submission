# XBCI 情绪强度分类

## 安装
```bash
pip install -r requirements.txt
```

## 环境
- Python 3.9+
- PyTorch 和 numpy（默认使用 CPU，可选 GPU ）

## 推理
```bash
python main.py --input <测试数据文件夹> --output <结果输出路径> --model model.pth
```