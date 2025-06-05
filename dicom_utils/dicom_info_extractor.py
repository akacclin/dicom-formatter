import pydicom
import pandas as pd
from pydicom.errors import InvalidDicomError
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_patient_info(ds: pydicom.dataset.FileDataset) -> Dict[str, Any]:
    """
    从DICOM数据集中提取患者信息
    
    Args:
        ds: DICOM数据集
    
    Returns:
        Dict[str, Any]: 包含患者信息的字典
    """
    # 处理患者姓名，可能是复杂的DICOM名字对象
    name_obj = ds.get("PatientName", "N/A")
    age_obj = ds.get("PatientAge", "N/A")
    name = str(name_obj).replace('^', '').strip()
    age = str(age_obj).replace('Y', '').strip()
    
    # 提取并返回基本信息
    return {
        "PID": ds.get("PatientID", "N/A"),
        "拼音": name,
        "扫描日期": ds.get("StudyDate", "N/A"),
        "年龄": age,
        "性别": ds.get("PatientSex", "N/A"),
        "诊断部位": ds.get("BodyPartExamined", "N/A")
    }


def find_first_dicom_in_dir(directory: Path) -> Optional[Path]:
    """
    在目录中找到第一个DICOM文件
    
    Args:
        directory: 要搜索的目录
    
    Returns:
        Optional[Path]: 第一个DICOM文件的路径，如果没找到则返回None
    """
    # 查找.dcm扩展名的文件
    for file_path in directory.glob("*.dcm"):
        return file_path
            
    return None


def process_directory(directory: Path) -> Optional[Dict[str, Any]]:
    """
    处理单个目录，提取第一个有效DICOM文件的信息
    
    Args:
        directory: 要处理的目录
    
    Returns:
        Optional[Dict[str, Any]]: 包含患者信息的字典，如果没有找到DICOM文件则返回None
    """
    if not directory.is_dir():
        return None
        
    # 找到目录中第一个DICOM文件
    dicom_file = find_first_dicom_in_dir(directory)
    if not dicom_file:
        return None
        
    # 读取DICOM文件
    ds = pydicom.dcmread(str(dicom_file), stop_before_pixels=True)
    if not ds:
        return None
        
    # 提取患者信息
    patient_info = extract_patient_info(ds)
    
    logger.debug(f"成功从 {directory} 读取DICOM信息: {patient_info['PID']} - {patient_info['拼音']}")
    return patient_info


def scan_directories_recursive(root_path: Path, max_depth: int = 5) -> List[Dict[str, Any]]:
    """
    递归扫描目录树，找到所有包含DICOM文件的目录并提取信息
    
    Args:
        root_path: 根目录路径
        max_depth: 最大递归深度，防止过深的目录结构
    
    Returns:
        List[Dict[str, Any]]: 所有提取到的患者信息列表
    """
    all_patient_info = []
    processed_dirs = set()
    
    def scan_dir(current_path: Path, depth: int = 0):
        # 防止无限递归和过深的目录结构
        if depth > max_depth:
            return
            
        # 跳过已处理的目录
        if str(current_path) in processed_dirs:
            return
            
        # 处理当前目录
        if current_path.is_dir():
            # 先检查当前目录是否有DICOM文件
            patient_info = process_directory(current_path)
            if patient_info:
                all_patient_info.append(patient_info)
                processed_dirs.add(str(current_path))
                # 如果已找到DICOM文件，不需要再深入此目录
                return
                
            # 如果当前目录没有DICOM文件，递归检查子目录
            for child_path in current_path.iterdir():
                if child_path.is_dir():
                    scan_dir(child_path, depth + 1)
    
    # 开始递归扫描
    logger.info(f"开始扫描根目录: {root_path}")
    scan_dir(root_path)
    
    logger.info(f"扫描完成，共找到 {len(all_patient_info)} 条患者记录")
    return all_patient_info


def save_to_excel(data: List[Dict[str, Any]], output_path: Path) -> None:
    """
    将提取的数据保存到Excel文件
    
    Args:
        data: 要保存的数据列表
        output_path: 输出Excel文件路径
    """
    if not data:
        logger.warning("没有数据可保存")
        return
        
    # 创建DataFrame并排序列
    df = pd.DataFrame(data)
    
    # 确保列的顺序一致
    columns = ["PID", "拼音", "扫描日期", "年龄", "性别", "诊断部位"]
    df = df.reindex(columns=[col for col in columns if col in df.columns])
    
    # 去除重复记录
    df_unique = df.drop_duplicates()
    logger.info(f"共有 {len(data)} 条记录，去重后剩余 {len(df_unique)} 条")
    
    # 添加空的姓名列作为第一列
    df_unique.insert(0, "姓名", "")
    
    # 确保输出目录存在
    output_dir = output_path.parent
    if output_dir.name and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        
    # 保存到Excel
    try:
        df_unique.to_excel(output_path, index=False, engine='openpyxl')
        logger.info(f"成功将数据保存到Excel文件: {output_path}")
    except Exception as e:
        logger.error(f"保存Excel文件时出错: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="从DICOM文件夹提取信息并生成Excel详情记录文件")
    parser.add_argument("input_dir", help="包含DICOM文件的根目录路径")
    parser.add_argument("output_file", help="要生成的Excel文件的完整路径（例如：output/summary.xlsx）")
    parser.add_argument("--max-depth", type=int, default=10, help="最大目录扫描深度，默认为10")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志信息")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 转换路径为Path对象
    input_path = Path(args.input_dir)
    output_path = Path(args.output_file)
    
    # 验证输入目录
    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"错误: 输入目录 '{input_path}' 不存在或不是一个有效的目录")
        return
    
    # 扫描目录并提取DICOM信息
    patient_data = scan_directories_recursive(input_path, args.max_depth)
    
    # 保存结果到Excel
    if patient_data:
        save_to_excel(patient_data, output_path)
    else:
        logger.warning("未找到任何DICOM数据，不生成Excel文件")


if __name__ == "__main__":
    main()