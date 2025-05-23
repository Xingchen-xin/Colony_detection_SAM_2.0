# colony_analysis/core/sam_model.py
from .detection import ColonyDetector
from .sam_model import SAMModel
import os
import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Union

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


class SAMModel:
    """统一的SAM模型封装类，集成了之前SAMSegmenter和SAMWrapper的功能"""

    def __init__(self, model_type="vit_b", checkpoint_path=None, config=None):
        """
        初始化SAM模型
        
        Args:
            model_type: 模型类型 ("vit_h", "vit_l", "vit_b")
            checkpoint_path: 模型权重路径
            config: 配置管理器或配置字典
        """
        self.model_type = model_type
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = self._resolve_checkpoint_path(
            checkpoint_path, model_type)

        # 获取SAM参数
        self.params = self._extract_sam_params(config)

        # 加载模型
        self._load_model()

        logging.info(f"SAM模型已初始化 ({model_type})，设备: {self.device}")

    def _resolve_checkpoint_path(self, checkpoint_path: Optional[str], model_type: str) -> str:
        """解析检查点路径"""
        if checkpoint_path and os.path.exists(checkpoint_path):
            return checkpoint_path

        # 默认路径映射
        default_paths = {
            "vit_h": "models/sam_vit_h_4b8939.pth",
            "vit_l": "models/sam_vit_l_0b3195.pth",
            "vit_b": "models/sam_vit_b_01ec64.pth"
        }

        path = default_paths.get(model_type)

        # 检查不同可能的位置
        possible_paths = [
            path,
            f"src/{path}",
            f"../{path}",
            Path.home() / ".colony_analysis" / "models" / Path(path).name
        ]

        for p in possible_paths:
            if os.path.exists(p):
                return str(p)

        raise FileNotFoundError(f"找不到模型文件。请下载 {model_type} 模型到: {path}")

    def _extract_sam_params(self, config) -> dict:
        """从配置中提取SAM参数"""
        default_params = {
            "points_per_side": 64,
            "pred_iou_thresh": 0.85,
            "stability_score_thresh": 0.8,
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": 1500,
        }

        if config is None:
            return default_params

        # 尝试从配置获取SAM参数
        try:
            if hasattr(config, 'get') and callable(config.get):
                sam_config = config.get('sam', {})
                if isinstance(sam_config, dict):
                    default_params.update(sam_config)
            elif isinstance(config, dict) and 'sam' in config:
                default_params.update(config['sam'])
        except Exception as e:
            logging.warning(f"获取SAM配置参数失败: {e}")

        return default_params

    def _load_model(self):
        """加载SAM模型和初始化组件"""
        try:
            # 加载SAM模型
            self.sam = sam_model_registry[self.model_type](
                checkpoint=self.checkpoint_path)
            self.sam.to(device=self.device)

            # 初始化预测器
            self.predictor = SamPredictor(self.sam)

            # 初始化自动掩码生成器
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                **self.params
            )

            logging.info(f"SAM模型加载成功，参数: {self.params}")

        except Exception as e:
            logging.error(f"加载SAM模型失败: {e}")
            raise

    def segment_with_prompts(self, image: np.ndarray,
                             points: Optional[List[List[float]]] = None,
                             point_labels: Optional[List[int]] = None,
                             boxes: Optional[np.ndarray] = None,
                             mask_input: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        使用提示进行分割
        
        Args:
            image: RGB格式图像
            points: 点坐标列表 [[x1,y1], [x2,y2], ...]
            point_labels: 点标签列表 [1, 0, ...] (1=前景, 0=背景)
            boxes: 边界框 [x1,y1,x2,y2]
            mask_input: 输入掩码
            
        Returns:
            (mask, score): 最佳分割掩码和得分
        """
        # 预处理图像
        image = self._preprocess_image(image)

        # 设置图像
        self.predictor.set_image(image)

        # 准备输入
        point_coords = np.array(points) if points else None
        point_labels_array = np.array(point_labels) if point_labels else None

        # 执行分割
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels_array,
            box=boxes,
            mask_input=mask_input,
            multimask_output=True
        )

        # 选择最佳掩码
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]

    def segment_everything(self, image: np.ndarray,
                           min_area: int = 25,
                           max_area: Optional[int] = None) -> Tuple[List[np.ndarray], List[float]]:
        """
        自动分割图像中的所有区域
        
        Args:
            image: RGB格式图像
            min_area: 最小区域面积
            max_area: 最大区域面积 (None表示无限制)
            
        Returns:
            (masks, scores): 掩码列表和对应得分列表
        """
        # 预处理图像
        image = self._preprocess_image(image)

        # 生成掩码
        masks_data = self.mask_generator.generate(image)

        # 过滤和提取结果
        masks = []
        scores = []

        for mask_data in masks_data:
            mask = mask_data['segmentation']
            score = mask_data['stability_score']
            area = mask_data['area']

            # 面积过滤
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue

            masks.append(mask)
            scores.append(score)

        return masks, scores

    def segment_grid(self, image: np.ndarray,
                     rows: int = 8, cols: int = 12,
                     padding: float = 0.05) -> Tuple[List[np.ndarray], List[str]]:
        """
        使用网格策略分割规则布局(如96孔板)
        
        Args:
            image: RGB格式图像
            rows: 行数
            cols: 列数
            padding: 网格边距比例
            
        Returns:
            (masks, labels): 掩码列表和对应标签列表
        """
        # 预处理图像
        image = self._preprocess_image(image)

        height, width = image.shape[:2]
        cell_height = height / rows
        cell_width = width / cols

        masks = []
        labels = []

        # 生成行标签 A-H
        row_labels = [chr(65 + i) for i in range(rows)]

        # 遍历每个网格单元
        for r in range(rows):
            for c in range(cols):
                # 计算单元格边界(添加padding)
                pad_y = int(cell_height * padding)
                pad_x = int(cell_width * padding)

                y1 = int(r * cell_height) + pad_y
                y2 = int((r + 1) * cell_height) - pad_y
                x1 = int(c * cell_width) + pad_x
                x2 = int((c + 1) * cell_width) - pad_x

                # 创建边界框
                box = np.array([x1, y1, x2, y2])

                try:
                    # 使用SAM分割该区域
                    mask, score = self.segment_with_prompts(image, boxes=box)
                    masks.append(mask)
                    labels.append(f"{row_labels[r]}{c+1}")
                except Exception as e:
                    logging.warning(f"分割网格单元 {row_labels[r]}{c+1} 失败: {e}")
                    # 创建空掩码避免程序中断
                    empty_mask = np.zeros((height, width), dtype=bool)
                    masks.append(empty_mask)
                    labels.append(f"{row_labels[r]}{c+1}")

        return masks, labels

    def find_diffusion_zone(self, image: np.ndarray,
                            colony_mask: np.ndarray,
                            expansion_pixels: int = 15) -> np.ndarray:
        """
        寻找菌落的扩散区域
        
        Args:
            image: RGB格式图像
            colony_mask: 菌落掩码
            expansion_pixels: 扩展像素数
            
        Returns:
            扩散区域掩码
        """
        # 膨胀操作创建扩散区域
        kernel = np.ones((3, 3), np.uint8)
        iterations = max(1, expansion_pixels // 3)

        expanded_mask = cv2.dilate(
            colony_mask.astype(np.uint8),
            kernel,
            iterations=iterations
        )

        # 扩散区域 = 膨胀区域 - 原始菌落区域
        diffusion_mask = expanded_mask - colony_mask.astype(np.uint8)

        return diffusion_mask > 0

    def update_params(self, new_params: dict):
        """更新SAM参数"""
        self.params.update(new_params)

        # 重新初始化掩码生成器
        try:
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                **self.params
            )
            logging.info(f"SAM参数已更新: {new_params}")
        except Exception as e:
            logging.error(f"更新SAM参数失败: {e}")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像，确保格式正确"""
        # 确保是RGB格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        return image

    @property
    def is_ready(self) -> bool:
        """检查模型是否准备就绪"""
        return hasattr(self, 'sam') and hasattr(self, 'predictor') and hasattr(self, 'mask_generator')


# colony_analysis/core/__init__.py

__all__ = ['SAMModel', 'ColonyDetector']
