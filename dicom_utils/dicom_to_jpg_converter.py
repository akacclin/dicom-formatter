from pathlib import Path
import pydicom
import argparse
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
from PIL import Image
import multiprocessing
import logging
import sys
import subprocess
import os
from typing import List, Dict, Any, Tuple, Optional, Callable


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# 配置诊断序列
DIAGNOSIS_SEQUENCES = {
    'HEAD': {
        'include': ['t1_gre_fsp_3d_sag_iso', 't2_fse_tra', 't2_fse_flair_tra_fs', 'tof3d_tra_uCS'],
        'exclude': ['NDC', 'MIP']
    },
    'ABDOMEN': {
        'include': ['dki_microV_cor_kidney', 'dwi_tra_IVIM_trig', 'dwi_tra_trig', 
                   't1_quick3d_tra_dualecho_bh', 't2_fse_tra_fs_trig', 't2_ssfse_cor_bh'],
        'exclude': []
    }
}


def find_latest_sequences(sequence_paths: List[Path]) -> List[Path]:
    """
    对于重复的扫描序列，保留最新的一个（根据序列名后缀判断）
    
    Args:
        sequence_paths: 序列路径列表
    
    Returns:
        List[Path]: 过滤后的序列路径列表
    """
    if not sequence_paths:
        return []
        
    filtered_sequences: Dict[str, Path] = {}
    
    for seq_path in sequence_paths:
        try:
            parts = seq_path.name.split('_')
            if not parts or not parts[-1].isdigit():
                continue
                
            prefix = '_'.join(parts[:-1])
            suffix = int(parts[-1])
            
            if prefix not in filtered_sequences or int(filtered_sequences[prefix].name.split('_')[-1]) < suffix:
                filtered_sequences[prefix] = seq_path
        except (ValueError, IndexError) as e:
            logger.warning(f"处理序列路径时出错 {seq_path.name}: {e}")
            continue
            
    return list(filtered_sequences.values())


def get_dicom_tag(directory: Path, tag_name: str) -> Optional[str]:
    """
    从目录中的第一个DICOM文件读取指定的标签值
    
    Args:
        directory: 包含DICOM文件的目录
        tag_name: 要读取的DICOM标签名
    
    Returns:
        Optional[str]: 标签值，如果读取失败则返回None
    """
    try:
        for path in directory.rglob('*.dcm'):
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                tag_value = ds.get(tag_name, None)
                if tag_value is not None:
                    return str(tag_value)
            except Exception:
                continue
                
        logger.warning(f"在目录 {directory} 中未找到包含标签 {tag_name} 的DICOM文件")
        return None
    except Exception as e:
        logger.error(f"读取目录 {directory} 中的DICOM标签时出错: {e}")
        return None


def select_sequences(body_part: str, directory: Path) -> List[Path]:
    """
    根据身体部位选择合适的诊断序列
    
    Args:
        body_part: 身体部位，例如'HEAD'或'ABDOMEN'
        directory: 包含序列文件夹的目录
    
    Returns:
        List[Path]: 符合条件的序列路径列表
    """
    if not directory.is_dir():
        logger.warning(f"{directory} 不是一个有效的目录")
        return []
        
    # 获取该身体部位的配置
    config = DIAGNOSIS_SEQUENCES.get(body_part.upper(), None)
    
    # 如果没有配置，返回所有子目录
    if config is None:
        logger.info(f"未找到 {body_part} 的配置，将使用所有序列")
        return [path for path in directory.iterdir() if path.is_dir()]
    
    include_patterns = config['include']
    exclude_patterns = config['exclude']
    
    filtered_sequences = []
    
    for seq_path in directory.iterdir():
        if not seq_path.is_dir():
            continue
            
        # 检查是否匹配排除模式
        if any(exclude_pattern in seq_path.name for exclude_pattern in exclude_patterns):
            continue
            
        # 检查是否匹配包含模式
        if any(seq_path.name.startswith(include_pattern) for include_pattern in include_patterns):
            filtered_sequences.append(seq_path)
    
    logger.info(f"为 {body_part} 找到 {len(filtered_sequences)} 个符合条件的序列")
    return filtered_sequences


def process_pixel_data(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    处理DICOM图像像素数据
    
    Args:
        ds: DICOM数据集
    
    Returns:
        np.ndarray: 处理后的8位像素数组
    """
    # 获取原始像素数据
    pixel_array = ds.pixel_array
    
    # 1. 应用模态LUT（Rescale Slope/Intercept）
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        rescale_slope = float(ds.RescaleSlope)
        rescale_intercept = float(ds.RescaleIntercept)
        pixel_array = pixel_array * rescale_slope + rescale_intercept
    else:
        # 尝试应用其他模态LUT
        try:
            pixel_array = apply_modality_lut(pixel_array, ds)
        except Exception as e:
            logger.debug(f"应用模态LUT时出错: {e}")

    # 2. 应用VOI LUT（窗宽窗位）
    try:
        if hasattr(ds, 'VOILUTFunction') and ds.VOILUTFunction == 'SIGMOID':
            pixel_array = apply_voi_lut(pixel_array, ds)
        elif hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            window_center = ds.WindowCenter
            window_width = ds.WindowWidth
            
            # 处理多值情况
            if hasattr(window_center, '__iter__') and not isinstance(window_center, str):
                window_center = float(window_center[0])
            else:
                window_center = float(window_center)
                
            if hasattr(window_width, '__iter__') and not isinstance(window_width, str):
                window_width = float(window_width[0])
            else:
                window_width = float(window_width)
                
            # 应用线性精确窗位窗宽转换
            pixel_array = apply_window_level(pixel_array, window_width, window_center)
        else:
            # 尝试使用默认VOI LUT设置
            pixel_array = apply_voi_lut(pixel_array, ds)
    except Exception as e:
        logger.debug(f"应用VOI LUT时出错: {e}")
        # 如果出错，尝试简单的归一化
        if pixel_array.max() != pixel_array.min():
            pixel_array = ((pixel_array - pixel_array.min()) / 
                         (pixel_array.max() - pixel_array.min()) * 255.0)
        
    # 3. 归一化到8位
    if pixel_array.max() != pixel_array.min():
        pixel_array = ((pixel_array - pixel_array.min()) / 
                     (pixel_array.max() - pixel_array.min()) * 255.0)
    else:
        pixel_array = np.zeros_like(pixel_array)
        
    # 4. 处理Photometric Interpretation
    if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = 255.0 - pixel_array
    
    # 转换为8位无符号整型
    return np.clip(pixel_array, 0, 255).astype(np.uint8)


def apply_window_level(data: np.ndarray, window: float, level: float) -> np.ndarray:
    """
    应用窗宽窗位进行线性精确转换
    
    Args:
        data: 输入像素数据
        window: 窗宽
        level: 窗位(窗中心)
    
    Returns:
        np.ndarray: 转换后的像素数据
    """
    lower_bound = level - window / 2
    upper_bound = level + window / 2
    
    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min
    
    result = np.copy(data)
    
    # 低于下界的值设为最小值
    mask_low = result <= lower_bound
    result[mask_low] = data_min
    
    # 高于上界的值设为最大值
    mask_high = result > upper_bound
    result[mask_high] = data_max
    
    # 位于窗口内的值进行线性映射
    mask_window = ~(mask_low | mask_high)
    if np.any(mask_window) and window > 0:
        result[mask_window] = (((result[mask_window] - lower_bound) / window) * data_range) + data_min
    
    return result


def convert_dicom_to_jpg(dicom_path: Path, output_dir: Path, resize_factor: float = 1, quality: int = 90) -> Optional[Path]:
    """
    将单个DICOM文件转换为JPG图像
    
    Args:
        dicom_path: DICOM文件路径
        output_dir: 输出目录
        resize_factor: 调整大小的比例因子
        quality: JPG质量(1-100)
    
    Returns:
        Optional[Path]: 保存的JPG文件路径，如果转换失败则返回None
    """
    if not dicom_path.exists() or dicom_path.suffix.lower() != '.dcm':
        return None
        
    try:
        # 读取DICOM文件
        ds = pydicom.dcmread(dicom_path)
        
        # 处理像素数据
        pixel_array = process_pixel_data(ds)
        
        # 创建图像对象
        image = Image.fromarray(pixel_array).convert('L')
        
        # 调整大小
        if resize_factor != 1.0 and resize_factor > 0:
            original_width, original_height = image.size
            new_width = int(original_width * resize_factor)
            new_height = int(original_height * resize_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存JPG文件
        output_path = output_dir / f"{dicom_path.stem}.jpg"
        image.save(output_path, "JPEG", quality=quality)
        
        return output_path
    except Exception as e:
        logger.error(f"转换DICOM文件 {dicom_path} 时出错: {e}")
        return None


def convert_sequence_directory(sequence_dir: Path, output_dir: Path, 
                           num_workers: int = None, resize_factor: float = 2/3) -> int:
    """
    转换序列目录中的所有DICOM文件为JPG
    
    Args:
        sequence_dir: 包含DICOM文件的序列目录
        output_dir: 输出目录
        num_workers: 并行处理的工作进程数，None表示使用所有可用CPU
        resize_factor: 调整大小的比例因子
    
    Returns:
        int: 成功转换的文件数量
    """
    if not sequence_dir.exists() or not sequence_dir.is_dir():
        logger.warning(f"序列目录不存在或不是有效目录: {sequence_dir}")
        return 0
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"转换序列: {sequence_dir.name} -> {output_dir}")
    
    # 收集所有DICOM文件
    dicom_files = list(sequence_dir.glob('*.dcm'))
    if not dicom_files:
        logger.warning(f"序列目录 {sequence_dir.name} 中未找到DICOM文件")
        return 0
    
    # 准备转换参数
    conversion_args = [(dicom_path, output_dir, resize_factor) for dicom_path in dicom_files]
    
    # 使用多进程并行转换
    successful_conversions = 0
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(convert_dicom_to_jpg, conversion_args)
        successful_conversions = sum(1 for result in results if result is not None)
    
    logger.info(f"序列 {sequence_dir.name}: 共 {len(dicom_files)} 个文件，成功转换 {successful_conversions} 个")
    return successful_conversions


def process_mra_to_mip(sequence_dir: Path, output_dir: Path) -> bool:
    """
    调用MRA_to_MIP.py处理MRA序列并生成MIP图像
    
    Args:
        sequence_dir: MRA序列目录
        output_dir: MIP输出目录
        
    Returns:
        bool: 处理是否成功
    """
    try:
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始处理MRA序列并生成MIP: {sequence_dir.name} -> {output_dir}")
        
        # 构建命令行参数
        cmd = [
            sys.executable,  # 当前Python解释器路径
            "MRA_to_MIP.py",
            sequence_dir.as_posix(),   # 输入路径
            "--output", output_dir.as_posix(),  # 输出路径
            "--axis", "Z",             # 默认使用Y轴旋转
            "--step", "10"             # 10度步长
        ]
        
        # 执行命令
        logger.info(f"执行命令: {' '.join(cmd)}")
        process = subprocess.run(
            cmd,
            # capture_output=True,
            text=True,
            check=False
        )
        
        # 检查结果
        if process.returncode == 0:
            logger.info(f"MRA->MIP转换成功，输出目录: {output_dir}")
            return True
        else:
            logger.error(f"MRA->MIP转换失败，返回码: {process.returncode}")
            logger.error(f"错误输出: {process.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"处理MRA->MIP时出错: {e}")
        return False


def process_subject_directory(subject_dir: Path, output_base_dir: Path, num_workers: int = None, disable_mip: bool = False) -> None:
    """
    处理受试者目录，提取和转换相关序列
    
    Args:
        subject_dir: 受试者目录
        output_base_dir: 输出基础目录
        num_workers: 并行处理的工作进程数
        disable_mip: 是否禁用MIP生成
    """
    if not subject_dir.exists() or not subject_dir.is_dir():
        logger.warning(f"受试者目录不存在或不是有效目录: {subject_dir}")
        return
    
    logger.info(f"开始处理受试者目录: {subject_dir.name}")
    
    # 获取身体部位和受试者ID
    body_part = get_dicom_tag(subject_dir, 'BodyPartExamined')
    subject_id = get_dicom_tag(subject_dir, 'PatientID')
    
    if body_part is None:
        logger.warning(f"在目录 {subject_dir} 中未找到身体部位信息，将使用'UNKNOWN'")
        body_part = 'UNKNOWN'
    
    if subject_id is None:
        logger.warning(f"在目录 {subject_dir} 中未找到受试者ID，将使用目录名")
        subject_id = subject_dir.name
    
    # 选择合适的序列
    sequences = select_sequences(body_part, subject_dir)
    if not sequences:
        logger.warning(f"在受试者目录 {subject_dir} 中未找到符合条件的序列")
        return
    
    # 对于重复的序列，只保留最新的
    sequences = find_latest_sequences(sequences)
    
    # 创建输出目录
    output_dir = output_base_dir / body_part / subject_id
    
    # 转换每个序列
    total_converted = 0
    for sequence_dir in sequences:
        # 普通DICOM到JPG的转换
        total_converted += convert_sequence_directory(
            sequence_dir, 
            output_dir / sequence_dir.name,
            num_workers=num_workers
        )
        
        # 对于tof3d_tra_uCS开头的序列，额外进行MIP转换（除非禁用）
        if not disable_mip and sequence_dir.name.startswith("tof3d_tra_uCS"):
            logger.info(f"检测到MRA序列: {sequence_dir.name}，将生成MIP图像")
            mip_output_dir = output_dir / f"{sequence_dir.name}_MIP"
            process_mra_to_mip(sequence_dir, mip_output_dir)
    
    logger.info(f"受试者 {subject_id}({body_part}) 处理完成，共转换 {total_converted} 个文件")


def main():
    """程序入口点"""
    parser = argparse.ArgumentParser(description="将DICOM文件转换为JPG图像")
    parser.add_argument('input_dir', type=str, help="包含DICOM文件的根目录路径")
    parser.add_argument('--output_dir', type=str, help="输出目录，默认与输入目录相同")
    parser.add_argument('--resize_factor', type=float, default=2/3, help="调整大小的比例因子，默认为2/3")
    parser.add_argument('--quality', type=int, default=90, help="JPG质量(1-100)，默认为90")
    parser.add_argument('--workers', type=int, default=None, help="并行工作进程数量，默认为CPU核心数")
    parser.add_argument('--verbose', '-v', action='store_true', help="显示详细日志")
    parser.add_argument('--disable-mip', action='store_true', help="禁用MRA序列的MIP图像生成")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 处理输入和输出目录
    input_path = Path(args.input_dir)
    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"输入目录不存在或不是有效目录: {input_path}")
        return
    
    output_path = Path(args.output_dir) if args.output_dir else input_path
    
    logger.info(f"开始从 {input_path} 转换DICOM文件到 {output_path}")
    
    # 处理根目录下的每个日期目录
    for date_dir in input_path.iterdir():
        if not date_dir.is_dir():
            continue
            
        logger.info(f"处理日期目录: {date_dir.name}")
        
        # 处理日期目录下的每个受试者目录
        for subject_dir in date_dir.iterdir():
            if subject_dir.is_dir():
                process_subject_directory(subject_dir, output_path, 
                                         num_workers=args.workers,
                                         disable_mip=args.disable_mip)
    
    logger.info("转换完成")


if __name__ == '__main__':
    main()
