# 配# colony_analysis/config/settings.py
from .settings import ConfigManager
import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class DetectionConfig:
    """检测配置"""
    model_type: str = 'vit_b'
    mode: str = 'auto'
    min_colony_area: int = 5000
    expand_pixels: int = 8
    merge_overlapping: bool = True
    use_preprocessing: bool = True
    overlap_threshold: float = 0.3


@dataclass
class SAMConfig:
    """SAM模型配置"""
    points_per_side: int = 64
    pred_iou_thresh: float = 0.85
    stability_score_thresh: float = 0.8
    min_mask_region_area: int = 1500
    crop_n_layers: int = 1
    crop_n_points_downscale_factor: int = 1


@dataclass
class AnalysisConfig:
    """分析配置"""
    advanced: bool = False
    learning_enabled: bool = False
    aerial_threshold: float = 0.6
    metabolite_threshold: float = 0.5
    enable_parallel: bool = False
    max_workers: int = 4


@dataclass
class OutputConfig:
    """输出配置"""
    debug: bool = False
    well_plate: bool = False
    rows: int = 8
    cols: int = 12
    save_masks: bool = True
    save_visualizations: bool = True
    image_format: str = 'jpg'


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = 'INFO'
    log_to_file: bool = True
    log_dir: Optional[str] = None
    max_log_files: int = 10


class ConfigManager:
    """改进的配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，None使用默认路径
        """
        self.config_path = self._resolve_config_path(config_path)

        # 初始化配置对象
        self.detection = DetectionConfig()
        self.sam = SAMConfig()
        self.analysis = AnalysisConfig()
        self.output = OutputConfig()
        self.logging = LoggingConfig()

        # 加载配置
        self._load_config()

        # 确保配置目录存在
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

    def _resolve_config_path(self, config_path: Optional[str]) -> str:
        """解析配置文件路径"""
        if config_path and os.path.exists(config_path):
            return config_path

        # 默认配置文件位置
        default_locations = [
            'config.yaml',
            'config.yml',
            'config.json',
            Path.home() / '.colony_analysis' / 'config.yaml',
            Path(__file__).parent.parent.parent / 'config.yaml'
        ]

        for path in default_locations:
            if os.path.exists(path):
                return str(path)

        # 如果没有找到，使用默认路径
        default_path = Path.home() / '.colony_analysis' / 'config.yaml'
        return str(default_path)

    def _load_config(self):
        """从文件加载配置"""
        if not os.path.exists(self.config_path):
            logging.info(f"配置文件不存在，使用默认配置: {self.config_path}")
            self.save_config()  # 保存默认配置
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.json'):
                    config_data = json.load(f)
                else:
                    config_data = yaml.safe_load(f) or {}

            # 更新配置对象
            self._update_config_from_dict(config_data)
            logging.info(f"已从 {self.config_path} 加载配置")

        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
            logging.info("使用默认配置")

    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """从字典更新配置"""
        for section_name, section_data in config_data.items():
            if not isinstance(section_data, dict):
                continue

            if hasattr(self, section_name):
                config_obj = getattr(self, section_name)

                # 更新dataclass字段
                for field_name, field_value in section_data.items():
                    if hasattr(config_obj, field_name):
                        try:
                            setattr(config_obj, field_name, field_value)
                        except Exception as e:
                            logging.warning(
                                f"设置配置 {section_name}.{field_name} 失败: {e}")

    def save_config(self):
        """保存配置到文件"""
        try:
            config_data = {
                'detection': asdict(self.detection),
                'sam': asdict(self.sam),
                'analysis': asdict(self.analysis),
                'output': asdict(self.output),
                'logging': asdict(self.logging)
            }

            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.endswith('.json'):
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                else:
                    yaml.dump(config_data, f,
                              default_flow_style=False, allow_unicode=True)

            logging.info(f"配置已保存到: {self.config_path}")

        except Exception as e:
            logging.error(f"保存配置文件失败: {e}")

    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            section: 配置节名称
            key: 配置键名称 (可选)
            default: 默认值
            
        Returns:
            配置值
        """
        if not hasattr(self, section):
            return default

        config_obj = getattr(self, section)

        if key is None:
            return config_obj

        return getattr(config_obj, key, default)

    def set(self, section: str, key: str, value: Any):
        """
        设置配置值
        
        Args:
            section: 配置节名称
            key: 配置键名称
            value: 配置值
        """
        if not hasattr(self, section):
            logging.warning(f"配置节不存在: {section}")
            return

        config_obj = getattr(self, section)

        if not hasattr(config_obj, key):
            logging.warning(f"配置键不存在: {section}.{key}")
            return

        try:
            setattr(config_obj, key, value)
            logging.debug(f"设置配置: {section}.{key} = {value}")
        except Exception as e:
            logging.error(f"设置配置失败: {e}")

    def update_from_args(self, args):
        """从命令行参数更新配置"""
        # 检测配置
        if hasattr(args, 'model') and args.model:
            self.detection.model_type = args.model
        if hasattr(args, 'mode') and args.mode:
            self.detection.mode = args.mode
        if hasattr(args, 'min_colony_area') and args.min_colony_area:
            self.detection.min_colony_area = args.min_colony_area
        if hasattr(args, 'expand_pixels') and args.expand_pixels:
            self.detection.expand_pixels = args.expand_pixels
        if hasattr(args, 'no_merge') and args.no_merge:
            self.detection.merge_overlapping = False
        if hasattr(args, 'no_preprocess') and args.no_preprocess:
            self.detection.use_preprocessing = False

        # 分析配置
        if hasattr(args, 'advanced') and args.advanced:
            self.analysis.advanced = True
        if hasattr(args, 'learn') and args.learn:
            self.analysis.learning_enabled = True

        # 输出配置
        if hasattr(args, 'debug') and args.debug:
            self.output.debug = True
        if hasattr(args, 'well_plate') and args.well_plate:
            self.output.well_plate = True
        if hasattr(args, 'rows') and args.rows:
            self.output.rows = args.rows
        if hasattr(args, 'cols') and args.cols:
            self.output.cols = args.cols

        # 日志配置
        if hasattr(args, 'verbose') and args.verbose:
            self.logging.level = 'DEBUG'

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'detection': asdict(self.detection),
            'sam': asdict(self.sam),
            'analysis': asdict(self.analysis),
            'output': asdict(self.output),
            'logging': asdict(self.logging)
        }

    def validate_config(self) -> bool:
        """验证配置的有效性"""
        try:
            # 验证检测配置
            assert self.detection.model_type in ['vit_b', 'vit_l', 'vit_h']
            assert self.detection.mode in ['auto', 'grid', 'hybrid']
            assert self.detection.min_colony_area > 0
            assert 0 <= self.detection.overlap_threshold <= 1

            # 验证SAM配置
            assert self.sam.points_per_side > 0
            assert 0 <= self.sam.pred_iou_thresh <= 1
            assert 0 <= self.sam.stability_score_thresh <= 1
            assert self.sam.min_mask_region_area > 0

            # 验证输出配置
            assert self.output.rows > 0
            assert self.output.cols > 0
            assert self.output.image_format in ['jpg', 'png', 'tiff']

            # 验证日志配置
            assert self.logging.level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']

            return True

        except AssertionError as e:
            logging.error(f"配置验证失败: {e}")
            return False
        except Exception as e:
            logging.error(f"配置验证错误: {e}")
            return False

    def reset_to_defaults(self):
        """重置为默认配置"""
        self.detection = DetectionConfig()
        self.sam = SAMConfig()
        self.analysis = AnalysisConfig()
        self.output = OutputConfig()
        self.logging = LoggingConfig()

        logging.info("配置已重置为默认值")

    def __repr__(self) -> str:
        """字符串表示"""
        return f"ConfigManager(config_path='{self.config_path}')"


# colony_analysis/config/__init__.py

__all__ = ['ConfigManager']


# 使用示例和测试
if __name__ == "__main__":
    # 测试配置管理器
    config = ConfigManager()

    # 测试配置验证
    assert config.validate_config()

    # 测试配置设置和获取
    config.set('detection', 'min_colony_area', 10000)
    assert config.get('detection', 'min_colony_area') == 10000

    # 测试保存和加载
    config.save_config()

    print("配置管理器测试通过!")
