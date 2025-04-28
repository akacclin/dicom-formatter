# DICOM格式化工具

本项目是一个用于将DICOM医疗影像转换为常见图像格式(JPG)的工具，并支持对MRA（磁共振血管造影）序列生成MIP（最大强度投影）图像。主要用于影像学习和预览，方便医生和研究人员快速查看DICOM序列而无需专业软件。
提供以下主要功能：
1. DICOM到JPG图像转换
2. MRA（磁共振血管造影）序列的MIP（最大强度投影）图像生成
3. DICOM信息提取和Excel报告生成

## 目录结构说明

```
dcm-formatter/
├── dicom_to_jpg_converter.py  # DICOM到JPG图像转换主脚本
├── MRA_to_MIP.py              # MRA序列MIP生成脚本
├── dicom_info_extractor.py    # DICOM信息提取脚本
├── main.py                    # 主入口脚本，整合所有功能
├── requirements.txt           # 依赖管理
└── README.md                  # 本说明文档
```

## 输入数据格式

项目期望的输入文件夹结构如下：

```
输入目录/
├── 20250410/                  # 日期文件夹
│   ├── name_20250410-xxxxxx-xxxx_xxxxxx/  # 受试者文件夹
│   │   ├── t1_gre_fsp_3d_sag_iso_1701/  # 序列1
│   │   │   ├── 1.dcm
│   │   │   ├── 2.dcm
│   │   │   └── ...
│   │   ├── t2_fse_tra_1702/   # 序列2
│   │   │   ├── 1.dcm
│   │   │   └── ...
│   │   └── tof3d_tra_uCS_1701/  # MRA序列
│   │       ├── 1.dcm
│   │       └── ...
│   └── 其他受试者/
└── 其他日期/
```
![image](https://github.com/user-attachments/assets/68d5a3d8-51f3-43fd-884c-870b3528a9a1)

## 输出格式

转换后的输出格式如下：

```
输出目录/
└── converted/
    └── HEAD/                  # 身体部位
        └── PID-20250410-xxxxxx-xxxx/  # 受试者ID
            ├── t1_gre_fsp_3d_sag_iso_1701/  # 序列1 JPG格式
            │   ├── 1.jpg
            │   └── ...
            ├── t2_fse_tra_1702/   # 序列2 JPG格式
            │   ├── 1.jpg
            │   └── ...
            ├── tof3d_tra_uCS_1701/  # MRA序列 JPG格式
            │   ├── 1.jpg
            │   └── ...
            └── tof3d_tra_uCS_1701_MIP/  # MIP图像
                ├── mip_000.jpg
                ├── mip_010.jpg
                └── ... # 不同角度的MIP图像
```

## 安装说明

### 环境需求

- Python 3.8+
- 支持CUDA的GPU（可选，但强烈推荐用于MIP生成）

### 使用UV安装

[UV](https://github.com/astral-sh/uv) 是一个快速、可靠的Python包管理工具。推荐使用UV安装依赖：

```bash
# 安装UV (如果尚未安装)
curl -sSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv venv
uv sync

# 激活虚拟环境
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

## 使用方法

### 基本用法

```bash
uv run main.py 输入目录路径 [选项]
```

### 参数说明

| 参数 | 描述 |
|------|------|
| `input_dir` | 必选，包含DICOM文件的根目录路径 |
| `--output_dir` | 可选，输出目录，默认与输入目录相同 |
| `--resize_factor` | 可选，调整大小的比例因子，默认为2/3 |
| `--quality` | 可选，JPG质量(1-100)，默认为90 |
| `--workers` | 可选，并行工作进程数量，默认使用所有CPU核心 |
| `--verbose, -v` | 可选，显示详细日志 |
| `--no_convert` | 可选，跳过DICOM到JPG的转换 |
| `--no_extract` | 可选，跳过DICOM信息提取 |
| `--disable-mip` | 可选，禁用MRA序列的MIP图像生成 |

### 示例

```bash
# 完整功能（转换+信息提取）
uv run main.py D:/DicomData

# 仅执行DICOM到JPG转换
uv run main.py D:/DicomData --no_extract

# 仅执行信息提取
uv run main.py D:/DicomData --no_convert

# 调整图像大小和质量
uv run main.py D:/DicomData --resize_factor 0.5 --quality 95

# 使用4个工作进程并输出详细日志
uv run main.py D:/DicomData --workers 4 --verbose

# 禁用MIP生成
uv run main.py D:/DicomData --disable-mip
```

## MIP图像生成说明

MIP（最大强度投影）图像生成是为MRA序列提供的特殊功能，它会生成多个角度的投影图像，有助于更好地观察血管结构。

**注意**：MIP图像生成需要大量计算资源，请注意：

- **使用GPU** (推荐): MIP生成会自动使用CUDA加速，大约20秒生成一组MIP图像
- **使用CPU**: 如果没有CUDA支持，进程会回退到CPU计算，可能需要两分钟
- 如果只需要普通DICOM转JPG功能，可使用`--disable-mip`选项禁用MIP生成

## MRA_to_MIP.py 独立使用

MRA_to_MIP脚本也可以独立使用，对单个MRA序列生成MIP图像：

```bash
uv run MRA_to_MIP.py 输入MRA序列目录 --output 输出目录 [选项]
```

### 参数

| 参数 | 描述 |
|------|------|
| `input` | 必选，包含DICOM文件的MRA序列目录 |
| `--output, -o` | 必选，保存生成的MIP图像的输出文件夹路径 |
| `--spacing, -s` | 可选，目标等向体素尺寸，单位为mm（不指定则使用原始间距的最小值） |
| `--axis, -a` | 可选，旋转轴: X (横向旋转), Y (纵向旋转), Z (轴向旋转)，默认为Z |
| `--step` | 可选，旋转角度步长，默认为10度 |
