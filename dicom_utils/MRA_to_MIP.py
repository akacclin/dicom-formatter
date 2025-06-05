import os
import pydicom
import numpy as np
import torch
import torch.nn.functional as F
import kornia.geometry.transform as T
from PIL import Image
import sys
import warnings
import traceback
import argparse
from pathlib import Path
from typing import Tuple, List, Optional, Union

def setup_device() -> torch.device:
    """设置运算设备并返回"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device

def resample_volume(volume_np: np.ndarray, 
                   original_spacing: List[float], 
                   target_spacing: List[float], 
                   device: Optional[torch.device] = None) -> torch.Tensor:
    """
    使用PyTorch将3D体数据(D, H, W)重采样到目标间距
    间距格式为(z, y, x)
    
    Args:
        volume_np: 输入体数据，形状为(D, H, W)
        original_spacing: 原始体素间距，格式为[z, y, x]，单位为mm
        target_spacing: 目标体素间距，格式为[z, y, x]，单位为mm
        device: 计算设备，默认自动选择
    
    Returns:
        重采样后的PyTorch张量，形状为(1, 1, D', H', W')
    """
    if device is None:
        device = setup_device()
    
    current_size = np.array(volume_np.shape)  # (D, H, W)
    original_spacing_np = np.array(original_spacing)  # (z, y, x)
    target_spacing_np = np.array(target_spacing)  # (z, y, x)

    # 根据原始大小和间距比例计算目标大小
    target_size = np.round(current_size * (original_spacing_np / target_spacing_np)).astype(int)
    target_size[target_size < 1] = 1  # 确保尺寸至少为1

    print(f"原始体积形状 (z, y, x): {volume_np.shape}")
    print(f"原始间距 (z, y, x): {original_spacing} mm")
    print(f"目标间距 (z, y, x): {target_spacing} mm")
    print(f"重采样后的目标形状 (z, y, x): {target_size}")

    # 将numpy转换为torch张量 (B=1, C=1, D, H, W) 并移至设备
    # 转换为float32以便PyTorch进行插值
    volume_ts = torch.from_numpy(volume_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    # 检查是否需要重采样（允许小差异）
    if np.allclose(current_size, target_size, atol=1):
        print("原始体积已接近目标大小，跳过重采样。")
        return volume_ts

    # 使用torch.nn.functional.interpolate进行重采样（3D使用三线性）
    try:
        resampled_volume_ts = F.interpolate(
            volume_ts,
            size=(int(target_size[0]), int(target_size[1]), int(target_size[2])),  # (D', H', W')元组
            mode='trilinear',
            align_corners=True
        )
        print(f"重采样成功，在{device}上的形状: {resampled_volume_ts.shape[2:]}")

    except Exception as e:
        print(f"重采样失败: {e}")
        traceback.print_exc()
        raise

    return resampled_volume_ts

def read_dicom_volume_with_info(folder_path: str) -> Tuple[np.ndarray, List[float]]:
    """
    从文件夹读取DICOM文件，排序并堆叠为3D numpy数组。
    返回体积数据(D, H, W)和间距(z, y, x)。
    
    Args:
        folder_path: DICOM文件夹路径
        
    Returns:
        体积数据和间距信息的元组
    """
    print(f"从以下位置读取DICOM文件: {folder_path}")
    folder = Path(folder_path)
    dicom_files = list(folder.glob('*.dcm'))

    if not dicom_files:
        raise FileNotFoundError(f"在{folder_path}中未找到.dcm文件")

    slices = []
    spacing_info = None  # 存储(SliceThickness, PixelSpacing[0], PixelSpacing[1]) 即 (z, y, x)
    unique_spacings = set()

    # 进度条显示
    try:
        from tqdm import tqdm
        file_iter = tqdm(dicom_files, desc="读取DICOM文件")
    except ImportError:
        file_iter = dicom_files
        print("提示：安装tqdm (pip install tqdm)可获得进度条显示。")

    for dcm_file in file_iter:
        try:
            ds = pydicom.dcmread(dcm_file)

            if not hasattr(ds, 'PixelData'):
                continue  # 跳过非图像文件

            slice_location = getattr(ds, 'SliceLocation', None)
            instance_number = getattr(ds, 'InstanceNumber', None)

            # 对切片进行排序：优先使用SliceLocation，备选InstanceNumber
            if slice_location is not None:
                slices.append((slice_location, ds))
            elif instance_number is not None:
                slices.append((instance_number, ds))
            else:
                warnings.warn(f"文件{dcm_file}缺少常见的排序标签(SliceLocation, InstanceNumber)，已跳过。")
                continue

            # 从找到的第一个有效切片中获取间距信息
            slice_thickness = getattr(ds, 'SliceThickness', None)
            pixel_spacing = getattr(ds, 'PixelSpacing', None)  # (row, col) -> (y, x)
            if slice_thickness is not None and pixel_spacing is not None and len(pixel_spacing) == 2:
                current_spacing = (float(slice_thickness), float(pixel_spacing[0]), float(pixel_spacing[1]))
                unique_spacings.add(tuple(current_spacing))
                if spacing_info is None:
                    spacing_info = list(current_spacing)

        except pydicom.errors.InvalidDicomError:
            continue  # 跳过无效的DICOM文件
        except Exception as e:
            print(f"读取文件{dcm_file}时出错: {e}")
            continue

    if not slices:
        raise FileNotFoundError(f"在{folder_path}中未找到带有排序标签的有效DICOM图像切片")

    slices.sort(key=lambda x: x[0])
    print(f"成功读取并排序了{len(slices)}个切片。")

    # 检查间距一致性
    if len(unique_spacings) > 1:
        print(f"警告：检测到{len(unique_spacings)}个不同的间距信息。使用首个找到的间距({spacing_info})进行重采样。建议检查DICOM一致性。")
    elif len(unique_spacings) == 0:
        warnings.warn("未找到间距信息。假设体素为各向同性，间距为[1, 1, 1]。这可能导致变形。")
        spacing_info = [1.0, 1.0, 1.0]
    else:
        if spacing_info:
            print(f"像素间距 (y, x): [{spacing_info[1]:.6f}, {spacing_info[2]:.6f}] mm")
            print(f"切片厚度 (z): {spacing_info[0]:.6f} mm")

    # 构建体积
    try:
        ref_ds = slices[0][1]
        slice_shape = (ref_ds.Rows, ref_ds.Columns)
        pixel_dtype = ref_ds.pixel_array.dtype

        pixel_arrays = []
        for i, (sort_key, ds) in enumerate(slices):
            # 检查尺寸和数据类型是否与第一个切片匹配，以便堆叠
            if ds.Rows == slice_shape[0] and ds.Columns == slice_shape[1] and ds.pixel_array.dtype == pixel_dtype:
                pixel_arrays.append(ds.pixel_array)
            else:
                # 可选：警告跳过的切片
                # warnings.warn(f"由于形状或数据类型不一致，跳过切片{getattr(ds, 'InstanceNumber', i)}。")
                pass

        if not pixel_arrays:
            raise ValueError("未找到用于堆叠的有效切片（检查一致性）。")

        volume_data = np.stack(pixel_arrays, axis=0)  # 沿z轴堆叠

    except Exception as e:
        raise RuntimeError(f"处理像素数据时出错: {e}")

    print(f"原始体积形状 (z, y, x): {volume_data.shape}")
    print(f"使用原始间距 (z, y, x): {spacing_info} mm")

    return volume_data, spacing_info

def generate_mip_rotations(volume_ts: torch.Tensor, 
                          output_folder: str, 
                          rotation_step_deg: int = 10, 
                          rotation_axis: str = 'Y') -> None:
    """
    使用GPU上的PyTorch生成180度旋转的MIP图像。
    假设volume_ts是正确设备上的(1, 1, D, H, W)张量，
    并且已经重采样为等向体素。
    
    Args:
        volume_ts: 体积数据张量，形状为(1, 1, D, H, W)
        output_folder: 输出文件夹路径
        rotation_step_deg: 旋转步长，单位为度
        rotation_axis: 旋转轴: 'X', 'Y', 或 'Z'
    """
    device = volume_ts.device
    print(f"使用设备生成MIP: {device}")

    _, _, D, H, W = volume_ts.shape
    print(f"用于旋转的体积形状 (D, H, W): ({D}, {H}, {W})")

    # 旋转角度 (0, 10, ..., 350度)
    angles_deg = torch.arange(0, 190, rotation_step_deg, device=device, dtype=volume_ts.dtype)

    # 旋转中心 (x, y, z) 其中x=W, y=H, z=D
    center = torch.tensor([[W/2, H/2, D/2]], device=device, dtype=volume_ts.dtype)  # 形状 (1, 3)

    # 创建输出目录
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
        angles_iter = tqdm(angles_deg, desc=f"生成MIP（围绕{rotation_axis}轴旋转）")
    except ImportError:
        angles_iter = angles_deg
        print("提示：安装tqdm (pip install tqdm)可获得进度条显示。")

    print(f"\n开始生成{len(angles_deg)}个MIP...")

    first_mip_max = None  # 用于一致的缩放

    for i, angle_deg in enumerate(angles_iter):
        angle_value = angle_deg.item()

        # 准备欧拉角（偏航、俯仰、滚转）基于rotation_axis
        # Kornia的rotate3d可能使用Z-Y-X欧拉角（偏航、俯仰、滚转）
        # yaw：围绕Z轴旋转
        # pitch：围绕Y轴旋转
        # roll：围绕X轴旋转
        yaw_batch = torch.tensor([0.0], device=device, dtype=volume_ts.dtype)
        pitch_batch = torch.tensor([0.0], device=device, dtype=volume_ts.dtype)
        roll_batch = torch.tensor([0.0], device=device, dtype=volume_ts.dtype)

        if rotation_axis.upper() == 'Y':
            # 围绕Y轴旋转（俯仰）
            pitch_batch = angle_deg.unsqueeze(0)
        elif rotation_axis.upper() == 'X':
            # 围绕X轴旋转（滚转）
            roll_batch = angle_deg.unsqueeze(0)
        elif rotation_axis.upper() == 'Z':
            # 围绕Z轴旋转（偏航）
            yaw_batch = angle_deg.unsqueeze(0)
        else:
            raise ValueError(f"不支持的rotation_axis: {rotation_axis}。请选择'X'、'Y'或'Z'。")

        try:
            # 使用欧拉角旋转体积（度）
            rotated_volume_ts = T.rotate3d(
                volume_ts,            # (1, 1, D, H, W)
                yaw_batch,            # (1,) - 围绕Z旋转（度）
                pitch_batch,          # (1,) - 围绕Y旋转（度）
                roll_batch,           # (1,) - 围绕X旋转（度）
                center=center,        # (1, 3) 格式为 (cx, cy, cz) = (W/2, H/2, D/2)
                mode='bilinear',      # 插值模式
                padding_mode='zeros'  # 用零填充外部区域
            )

            # 沿深度方向执行最大强度投影（dim 2）
            mip_ts, _ = torch.max(rotated_volume_ts, dim=2)  # 结果形状 (1, 1, H, W)

            # 转换为numpy数组 (H, W)
            mip_np = mip_ts.squeeze().cpu().numpy()

            # 缩放到0-255用于保存为8位图像
            if first_mip_max is None:
                first_mip_max = mip_np.max().item()
                if first_mip_max < 1e-6:
                    first_mip_max = 1.0
                    warnings.warn("第一个MIP最大值为零或接近零，按1.0缩放。")

            scaled_mip_np = (mip_np / first_mip_max) * 255.0
            scaled_mip_np = np.clip(scaled_mip_np, 0, 255).astype(np.uint8)

            # 保存图像
            img = Image.fromarray(scaled_mip_np)
            output_filename = output_path / f"mip_{int(angle_value):03d}.jpg"
            img.save(output_filename)

        except Exception as e:
            print(f"\n处理角度{angle_value}°时出错: {e}")
            traceback.print_exc()
            continue  # 继续下一个角度

    print("\nMIP图像生成完成。")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将MRA DICOM数据转换为MIP图像')
    parser.add_argument('input', type=str, help='包含DICOM文件的输入文件夹路径')
    parser.add_argument('--output', '-o', type=str, required=True, help='保存生成的MIP图像的输出文件夹路径')
    parser.add_argument('--spacing', '-s', type=float, help='目标等向体素尺寸，单位为mm（不指定则使用原始间距的最小值）')
    parser.add_argument('--axis', '-a', type=str, default='Z', choices=['X', 'Y', 'Z'], 
                        help='旋转轴: X (横向旋转), Y (纵向旋转), Z (轴向旋转)')
    parser.add_argument('--step', type=int, default=10, help='旋转角度步长（默认10度）')
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.isdir(args.input):
        print(f"错误：输入文件夹不存在或不是目录: {args.input}")
        sys.exit(1)
    
    device = setup_device()
        
    try:
        # 步骤1：读取DICOM文件并获取体积数据和间距
        volume_data_np, original_spacing = read_dicom_volume_with_info(args.input)

        # 确定目标间距
        if args.spacing is None:
            if original_spacing is None:
                raise ValueError("无法确定原始间距。请指定目标间距值或确保DICOM有间距标签。")
            min_spacing = min(original_spacing)
            target_spacing_xyz = [min_spacing, min_spacing, min_spacing]
            print(f"未指定目标间距值，使用原始间距的最小值{min_spacing:.3f} mm作为目标。")
        else:
            target_spacing_xyz = [args.spacing, args.spacing, args.spacing]
            print(f"使用指定的目标间距{args.spacing:.3f} mm进行重采样。")

        # 步骤2：重采样为等向体素（对正确旋转至关重要）
        volume_tensor_resampled = resample_volume(volume_data_np, original_spacing, target_spacing_xyz, device)

        # 步骤3：生成并保存旋转的MIP图像
        generate_mip_rotations(volume_tensor_resampled, args.output, 
                              rotation_step_deg=args.step, rotation_axis=args.axis)

        print(f"所有生成的MIP图像已保存到文件夹: {args.output}")

    except Exception as e:
        print(f"发生意外错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()