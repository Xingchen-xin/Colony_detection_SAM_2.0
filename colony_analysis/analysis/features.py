# colony_analysis/analysis/features.py
import cv2
import numpy as np
from typing import Dict, Any, Optional


class FeatureExtractor:
    """菌落特征提取器"""

    def __init__(self, extractor_type: str = 'basic'):
        """
        初始化特征提取器
        
        Args:
            extractor_type: 提取器类型 ('basic', 'aerial', 'metabolite')
        """
        self.extractor_type = extractor_type

    def extract(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        提取特征
        
        Args:
            img: 菌落图像
            mask: 菌落掩码
            
        Returns:
            特征字典
        """
        if self.extractor_type == 'basic':
            return self._extract_basic_features(img, mask)
        elif self.extractor_type == 'aerial':
            return self._extract_aerial_features(img, mask)
        elif self.extractor_type == 'metabolite':
            return self._extract_metabolite_features(img, mask)
        else:
            return {}

    def _extract_basic_features(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """提取基本形态特征"""
        # 确保掩码是二值图像
        binary_mask = mask > 0

        # 计算面积
        area = np.sum(binary_mask)

        # 获取轮廓
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        features = {'area': float(area)}

        # 如果找到轮廓
        if contours:
            # 获取最大轮廓
            contour = max(contours, key=cv2.contourArea)

            # 计算周长
            perimeter = cv2.arcLength(contour, True)
            features['perimeter'] = float(perimeter)

            # 计算圆形度: 4π × 面积 / 周长²
            if perimeter > 0:
                circularity = (4 * np.pi * cv2.contourArea(contour)
                               ) / (perimeter * perimeter)
                features['circularity'] = float(circularity)

            # 计算最小外接矩形和长宽比
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                features['aspect_ratio'] = float(aspect_ratio)

            # 计算凸性
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                convexity = cv2.contourArea(contour) / hull_area
                features['convexity'] = float(convexity)

        # 计算边缘密度
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(
            img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges[binary_mask]) / area if area > 0 else 0
        features['edge_density'] = float(edge_density)

        return features

    def _extract_aerial_features(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """提取气生菌丝特征"""
        # 确保掩码是二值图像
        binary_mask = mask > 0

        # 转换到HSV空间
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        _, s, v = cv2.split(hsv)

        # 使用高亮度和低饱和度特征识别气生菌丝
        aerial_mask = (v > 200) & (s < 50) & binary_mask
        aerial_ratio = np.sum(
            aerial_mask) / np.sum(binary_mask) if np.sum(binary_mask) > 0 else 0

        # 计算气生菌丝"高度"
        if np.sum(aerial_mask) > 0:
            aerial_height_mean = np.mean(v[aerial_mask])
            aerial_height_std = np.std(v[aerial_mask])
            aerial_height_max = np.max(v[aerial_mask])
        else:
            aerial_height_mean = 0
            aerial_height_std = 0
            aerial_height_max = 0

        # 返回特征
        return {
            'morphology_aerial_area': float(np.sum(aerial_mask)),
            'morphology_aerial_ratio': float(aerial_ratio),
            'morphology_aerial_height_mean': float(aerial_height_mean),
            'morphology_aerial_height_std': float(aerial_height_std),
            'morphology_aerial_height_max': float(aerial_height_max)
        }

    def _extract_metabolite_features(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """提取代谢产物特征"""
        # 确保掩码是二值图像
        binary_mask = mask > 0

        # 提取RGB通道
        r_channel = img[:, :, 0]
        g_channel = img[:, :, 1]
        b_channel = img[:, :, 2]

        # 检测蓝色素(Actinorhodin)
        blue_mask = (b_channel > 100) & (b_channel > r_channel +
                                         20) & (b_channel > g_channel + 20) & binary_mask
        blue_area = np.sum(blue_mask)
        blue_ratio = blue_area / \
            np.sum(binary_mask) if np.sum(binary_mask) > 0 else 0

        # 检测红色素(Prodigiosin)
        red_mask = (r_channel > 100) & (r_channel > b_channel +
                                        20) & (r_channel > g_channel + 20) & binary_mask
        red_area = np.sum(red_mask)
        red_ratio = red_area / \
            np.sum(binary_mask) if np.sum(binary_mask) > 0 else 0

        # 计算色素强度
        blue_intensity = np.mean(b_channel[blue_mask]) if blue_area > 0 else 0
        red_intensity = np.mean(r_channel[red_mask]) if red_area > 0 else 0

        # 返回特征
        return {
            'metabolite_blue_area': float(blue_area),
            'metabolite_blue_ratio': float(blue_ratio),
            'metabolite_has_blue_pigment': blue_ratio > 0.05,
            'metabolite_blue_intensity_mean': float(blue_intensity),
            'metabolite_red_area': float(red_area),
            'metabolite_red_ratio': float(red_ratio),
            'metabolite_has_red_pigment': red_ratio > 0.05,
            'metabolite_red_intensity_mean': float(red_intensity)
        }


def extract_advanced_features(img: np.ndarray, mask: np.ndarray, diffusion_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    提取高级特征
    
    Args:
        img: 菌落图像
        mask: 菌落掩码
        diffusion_mask: 扩散区域掩码
        
    Returns:
        高级特征字典
    """
    features = {}

    # 分析代谢产物扩散
    if diffusion_mask is not None:
        diffusion_features = _analyze_diffusion(img, mask, diffusion_mask)
        features['diffusion'] = diffusion_features

    # 分析菌落内部区域
    internal_features = _analyze_internal_structure(img, mask)
    features['internal'] = internal_features

    return features


def _analyze_diffusion(img: np.ndarray, mask: np.ndarray, diffusion_mask: np.ndarray) -> Dict[str, Any]:
    """分析扩散区域"""
    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]

    # 蓝色素扩散
    blue_diffusion = (b_channel > 100) & (
        b_channel > r_channel + 20) & (b_channel > g_channel + 20) & diffusion_mask
    blue_diffusion_area = np.sum(blue_diffusion)

    # 红色素扩散
    red_diffusion = (r_channel > 100) & (
        r_channel > b_channel + 20) & (r_channel > g_channel + 20) & diffusion_mask
    red_diffusion_area = np.sum(red_diffusion)

    # 计算扩散比例
    diffusion_area = np.sum(diffusion_mask)
    colony_area = np.sum(mask)

    return {
        'blue_area': float(blue_diffusion_area),
        'red_area': float(red_diffusion_area),
        'blue_ratio': float(blue_diffusion_area / diffusion_area) if diffusion_area > 0 else 0,
        'red_ratio': float(red_diffusion_area / diffusion_area) if diffusion_area > 0 else 0,
        'diffusion_colony_ratio': float(diffusion_area / colony_area) if colony_area > 0 else 0
    }


def _analyze_internal_structure(img: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """分析菌落内部结构"""
    # 转换到HSV空间
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # 创建距离变换图
    dist_transform = cv2.distanceTransform(
        mask.astype(np.uint8), cv2.DIST_L2, 5)
    max_dist = np.max(dist_transform) if np.max(dist_transform) > 0 else 1
    normalized_dist = dist_transform / max_dist

    # 区分不同区域
    center_region = normalized_dist > 0.7
    middle_region = (normalized_dist > 0.3) & (normalized_dist <= 0.7)
    edge_region = (normalized_dist <= 0.3) & (mask > 0)

    # 分析各区域亮度和纹理
    center_brightness = np.mean(v[center_region]) if np.sum(
        center_region) > 0 else 0
    middle_brightness = np.mean(v[middle_region]) if np.sum(
        middle_region) > 0 else 0
    edge_brightness = np.mean(v[edge_region]) if np.sum(edge_region) > 0 else 0

    # 检测纹理变化
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(
        img.shape) == 3 else img
    texture = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.abs(texture)

    center_texture = np.mean(texture[center_region]) if np.sum(
        center_region) > 0 else 0
    middle_texture = np.mean(texture[middle_region]) if np.sum(
        middle_region) > 0 else 0
    edge_texture = np.mean(texture[edge_region]) if np.sum(
        edge_region) > 0 else 0

    return {
        'center_brightness': float(center_brightness),
        'middle_brightness': float(middle_brightness),
        'edge_brightness': float(edge_brightness),
        'center_texture': float(center_texture),
        'middle_texture': float(middle_texture),
        'edge_texture': float(edge_texture),
        'brightness_gradient': float(center_brightness - edge_brightness),
        'texture_gradient': float(center_texture - edge_texture)
    }
