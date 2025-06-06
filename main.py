import argparse
import os
from pathlib import Path
import logging
import sys
import shutil
from dicom_utils import dicom_to_jpg_converter, dicom_info_extractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_converted_folder(output_path: Path) -> bool:
    """
    检查是否存在converted文件夹
    返回True表示可以继续执行，False表示需要终止执行
    """
    converted_path = output_path
    if converted_path.exists():
        logger.warning(f"检测到已存在converted文件夹: {converted_path}")
        user_input = input("是否删除已存在的converted文件夹？(y/n): ").lower().strip()
        if user_input == 'y':
            shutil.rmtree(converted_path)
            logger.info("已删除existing converted文件夹")
            return True
        else:
            logger.info("用户选择保留existing converted文件夹，程序终止")
            return False
    return True

def main():
    """
    DICOM文件处理工具的主入口
    提供以下功能：
    1. DICOM文件转换为JPG图像
    2. 提取DICOM文件的患者信息到Excel
    3. 对MRA序列生成MIP（最大密度投影）图像
    """
    parser = argparse.ArgumentParser(description="DICOM文件处理工具 - 转换和信息提取")
    parser.add_argument('--input_dir', type=str, help="包含DICOM文件的根目录路径，默认为./data")
    parser.add_argument('--output_dir', type=str, help="输出目录，默认与输入目录相同")
    parser.add_argument('--resize_factor', type=float, default=1, help="调整大小的比例因子，默认为1")
    parser.add_argument('--quality', type=int, default=90, help="JPG质量(1-100)，默认为90")
    parser.add_argument('--workers', type=int, default=None, help="并行工作进程数量，默认为CPU核心数")
    parser.add_argument('--verbose', '-v', action='store_true', help="显示详细日志")
    parser.add_argument('--no_convert', action='store_true', help="跳过DICOM到JPG的转换")
    parser.add_argument('--no_extract', action='store_true', help="跳过DICOM信息提取")
    parser.add_argument('--disable-mip', action='store_true', help="禁用MRA序列的MIP图像生成")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 设置输入目录，默认为./data
    input_path = Path(args.input_dir) if args.input_dir else Path('data')
    if not input_path.exists():
        if not args.input_dir:  # 如果是默认的data目录不存在，则创建它
            logger.info(f"创建默认数据目录: {input_path}")
            input_path.mkdir(parents=True, exist_ok=True)
        else:
            logger.error(f"输入目录不存在: {input_path}")
            return
    elif not input_path.is_dir():
        logger.error(f"指定的路径不是有效目录: {input_path}")
        return
    
    # 检查converted文件夹
    if not check_converted_folder(output_path):
        return

    output_path = Path(args.output_dir) if args.output_dir else Path('Converted')
    
    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)



    # 1. 执行DICOM信息提取
    if not args.no_extract:
        logger.info("开始DICOM信息提取...")
        excel_output = output_path / 'dicom_info.xlsx'
        
        # 获取患者信息
        patient_data = dicom_info_extractor.scan_directories_recursive(input_path)
        
        # 保存到Excel文件
        if patient_data:
            dicom_info_extractor.save_to_excel(patient_data, excel_output)
            logger.info(f"DICOM信息提取完成，结果保存至: {excel_output}")
        else:
            logger.warning("未找到任何DICOM数据，不生成Excel文件")
    
    # 2. 执行DICOM到JPG的转换
    if not args.no_convert:
        logger.info("开始DICOM到JPG的转换...")
        # 构建转换器参数并调用
        converter_args = [
            str(input_path),
            '--output_dir', str(output_path),
            '--resize_factor', str(args.resize_factor),
            '--quality', str(args.quality)
        ]
        
        if args.workers:
            converter_args.extend(['--workers', str(args.workers)])
            
        if args.verbose:
            converter_args.append('--verbose')
            
        if args.disable_mip:
            converter_args.append('--disable-mip')
            
        # 临时替换sys.argv并调用转换器主函数
        old_argv = sys.argv
        sys.argv = ['dicom_to_jpg_converter.py'] + converter_args
        try:
            dicom_to_jpg_converter.main()
        finally:
            sys.argv = old_argv
            
        logger.info("DICOM到JPG的转换完成")
    
    logger.info("所有处理完成")

if __name__ == '__main__':
    main()
