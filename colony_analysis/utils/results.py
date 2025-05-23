# 结# colony_analysis/utils/results.py
from .validation import ImageValidator, DataValidator
from .visualization import Visualizer
from .results import ResultManager
from .logging import LogManager
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import cv2
import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class ResultManager:
    """统一的结果管理器"""

    def __init__(self, output_dir: str):
        """
        初始化结果管理器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.directories = self._create_directory_structure()

        logging.info(f"结果管理器初始化完成，输出目录: {self.output_dir}")

    def _create_directory_structure(self) -> Dict[str, Path]:
        """创建输出目录结构"""
        directories = {
            'root': self.output_dir,
            'results': self.output_dir / 'results',
            'colonies': self.output_dir / 'colonies',
            'masks': self.output_dir / 'masks',
            'visualizations': self.output_dir / 'visualizations',
            'debug': self.output_dir / 'debug',
            'reports': self.output_dir / 'reports'
        }

        # 创建所有目录
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        return directories

    def save_all_results(self, colonies: List[Dict], args) -> Dict[str, str]:
        """
        保存所有结果
        
        Args:
            colonies: 分析后的菌落列表
            args: 命令行参数
            
        Returns:
            保存的文件路径字典
        """
        saved_files = {}

        try:
            # 1. 保存CSV结果
            csv_path = self.save_csv_results(colonies)
            saved_files['csv'] = str(csv_path)

            # 2. 保存JSON结果（如果是高级分析）
            if getattr(args, 'advanced', False):
                json_path = self.save_json_results(colonies)
                saved_files['json'] = str(json_path)

            # 3. 保存菌落图像
            images_dir = self.save_colony_images(colonies)
            saved_files['images'] = str(images_dir)

            # 4. 保存掩码（如果启用调试）
            if getattr(args, 'debug', False):
                masks_dir = self.save_colony_masks(colonies)
                saved_files['masks'] = str(masks_dir)

            # 5. 生成分析报告
            report_path = self.generate_analysis_report(colonies, args)
            saved_files['report'] = str(report_path)

            logging.info(f"结果保存完成: {len(saved_files)} 个文件/目录")
            return saved_files

        except Exception as e:
            logging.error(f"保存结果时发生错误: {e}")
            raise

    def save_csv_results(self, colonies: List[Dict]) -> Path:
        """保存CSV格式的结果"""
        rows = []

        for colony in colonies:
            row = {
                'id': colony.get('id', 'unknown'),
                'well_position': colony.get('well_position', ''),
                'area': float(colony.get('area', 0)),
                'detection_method': colony.get('detection_method', 'unknown'),
                'sam_score': colony.get('sam_score', 0.0)
            }

            # 添加特征
            features = colony.get('features', {})
            for name, value in features.items():
                row[f'feature_{name}'] = self._safe_convert_value(value)

            # 添加评分
            scores = colony.get('scores', {})
            for name, value in scores.items():
                row[f'score_{name}'] = self._safe_convert_value(value)

            # 添加表型
            phenotype = colony.get('phenotype', {})
            for name, value in phenotype.items():
                row[f'phenotype_{name}'] = str(value)

            rows.append(row)

        # 保存CSV
        df = pd.DataFrame(rows)
        csv_path = self.directories['results'] / 'analysis_results.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')

        logging.info(f"CSV结果已保存: {csv_path}")
        return csv_path

    def save_json_results(self, colonies: List[Dict]) -> Path:
        """保存JSON格式的详细结果"""
        # 创建可序列化的数据
        serializable_data = []

        for colony in colonies:
            colony_data = {
                'id': colony.get('id', 'unknown'),
                'basic_info': {
                    'area': float(colony.get('area', 0)),
                    'centroid': colony.get('centroid', (0, 0)),
                    'bbox': colony.get('bbox', (0, 0, 0, 0)),
                    'detection_method': colony.get('detection_method', 'unknown')
                },
                'analysis_results': {
                    'features': self._serialize_dict(colony.get('features', {})),
                    'scores': self._serialize_dict(colony.get('scores', {})),
                    'phenotype': colony.get('phenotype', {}),
                    'advanced_features': self._serialize_dict(colony.get('advanced_features', {}))
                },
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'sam_score': colony.get('sam_score', 0.0)
                }
            }

            serializable_data.append(colony_data)

        # 保存JSON
        json_path = self.directories['results'] / 'detailed_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        logging.info(f"JSON结果已保存: {json_path}")
        return json_path

    def save_colony_images(self, colonies: List[Dict]) -> Path:
        """保存菌落图像"""
        import cv2

        images_saved = 0

        for colony in colonies:
            if 'img' not in colony:
                continue

            colony_id = colony.get(
                'well_position') or colony.get('id', 'unknown')

            # 保存原始菌落图像
            if 'img' in colony:
                img_path = self.directories['colonies'] / \
                    f"{colony_id}_original.jpg"
                cv2.imwrite(str(img_path), cv2.cvtColor(
                    colony['img'], cv2.COLOR_RGB2BGR))
                images_saved += 1

            # 保存掩码应用的图像
            if 'masked_img' in colony:
                masked_path = self.directories['colonies'] / \
                    f"{colony_id}_masked.jpg"
                cv2.imwrite(str(masked_path), cv2.cvtColor(
                    colony['masked_img'], cv2.COLOR_RGB2BGR))
                images_saved += 1

        logging.info(f"已保存 {images_saved} 个菌落图像")
        return self.directories['colonies']

    def save_colony_masks(self, colonies: List[Dict]) -> Path:
        """保存菌落掩码"""
        import cv2

        masks_saved = 0

        for colony in colonies:
            if 'mask' not in colony:
                continue

            colony_id = colony.get(
                'well_position') or colony.get('id', 'unknown')

            # 保存二值掩码
            mask_path = self.directories['masks'] / f"{colony_id}_mask.png"
            mask_img = (colony['mask'] * 255).astype(np.uint8)
            cv2.imwrite(str(mask_path), mask_img)
            masks_saved += 1

        logging.info(f"已保存 {masks_saved} 个菌落掩码")
        return self.directories['masks']

    def generate_analysis_report(self, colonies: List[Dict], args) -> Path:
        """生成分析报告"""
        report_data = {
            'analysis_summary': self._generate_summary_stats(colonies),
            'detection_info': self._generate_detection_info(colonies, args),
            'phenotype_distribution': self._generate_phenotype_stats(colonies),
            'quality_metrics': self._generate_quality_metrics(colonies)
        }

        # 保存报告
        report_path = self.directories['reports'] / 'analysis_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # 也生成文本格式的简要报告
        text_report_path = self._generate_text_report(report_data)

        logging.info(f"分析报告已生成: {report_path}")
        return report_path

    def _generate_summary_stats(self, colonies: List[Dict]) -> Dict:
        """生成统计摘要"""
        if not colonies:
            return {'total_colonies': 0}

        areas = [colony.get('area', 0) for colony in colonies]
        scores = [colony.get('sam_score', 0)
                  for colony in colonies if 'sam_score' in colony]

        return {
            'total_colonies': len(colonies),
            'area_stats': {
                'mean': float(np.mean(areas)),
                'median': float(np.median(areas)),
                'std': float(np.std(areas)),
                'min': float(np.min(areas)),
                'max': float(np.max(areas))
            },
            'detection_quality': {
                'mean_sam_score': float(np.mean(scores)) if scores else 0.0,
                'high_quality_colonies': len([s for s in scores if s > 0.9])
            }
        }

    def _generate_detection_info(self, colonies: List[Dict], args) -> Dict:
        """生成检测信息"""
        detection_methods = {}
        for colony in colonies:
            method = colony.get('detection_method', 'unknown')
            detection_methods[method] = detection_methods.get(method, 0) + 1

        return {
            'detection_mode': getattr(args, 'mode', 'unknown'),
            'model_type': getattr(args, 'model', 'unknown'),
            'detection_methods': detection_methods,
            'advanced_analysis': getattr(args, 'advanced', False)
        }

    def _generate_phenotype_stats(self, colonies: List[Dict]) -> Dict:
        """生成表型统计"""
        phenotype_stats = {}

        for colony in colonies:
            phenotype = colony.get('phenotype', {})
            for category, value in phenotype.items():
                if category not in phenotype_stats:
                    phenotype_stats[category] = {}
                phenotype_stats[category][value] = phenotype_stats[category].get(
                    value, 0) + 1

        return phenotype_stats

    def _generate_quality_metrics(self, colonies: List[Dict]) -> Dict:
        """生成质量指标"""
        total = len(colonies)
        if total == 0:
            return {}

        # 计算各种质量指标
        with_features = len([c for c in colonies if c.get('features')])
        with_scores = len([c for c in colonies if c.get('scores')])
        with_phenotype = len([c for c in colonies if c.get('phenotype')])

        return {
            'completeness': {
                'features_extracted': with_features / total,
                'scores_calculated': with_scores / total,
                'phenotype_classified': with_phenotype / total
            },
            'data_quality': {
                'total_colonies': total,
                'successful_analysis': with_features
            }
        }

    def _generate_text_report(self, report_data: Dict) -> Path:
        """生成文本格式的报告"""
        lines = []
        lines.append("=== 菌落分析报告 ===")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 基本统计
        summary = report_data.get('analysis_summary', {})
        lines.append(f"检测到的菌落总数: {summary.get('total_colonies', 0)}")

        if 'area_stats' in summary:
            area_stats = summary['area_stats']
            lines.append(f"平均面积: {area_stats['mean']:.2f}")
            lines.append(
                f"面积范围: {area_stats['min']:.2f} - {area_stats['max']:.2f}")

        # 检测信息
        detection_info = report_data.get('detection_info', {})
        lines.append(
            f"检测模式: {detection_info.get('detection_mode', 'unknown')}")
        lines.append(f"SAM模型: {detection_info.get('model_type', 'unknown')}")

        # 表型分布
        phenotype_dist = report_data.get('phenotype_distribution', {})
        if phenotype_dist:
            lines.append("\n=== 表型分布 ===")
            for category, distribution in phenotype_dist.items():
                lines.append(f"{category}:")
                for phenotype, count in distribution.items():
                    lines.append(f"  {phenotype}: {count}")

        # 保存文本报告
        text_path = self.directories['reports'] / 'analysis_report.txt'
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return text_path

    def _serialize_dict(self, data: Dict) -> Dict:
        """序列化字典中的numpy类型"""
        if not isinstance(data, dict):
            return data

        serialized = {}
        for key, value in data.items():
            serialized[key] = self._safe_convert_value(value)

        return serialized

    def _safe_convert_value(self, value: Any) -> Any:
        """安全转换值为可序列化类型"""
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.generic):
            return value.item()
        elif isinstance(value, (list, tuple)):
            return [self._safe_convert_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._safe_convert_value(v) for k, v in value.items()}
        else:
            return value


# colony_analysis/utils/visualization.py


class Visualizer:
    """统一的可视化工具"""

    def __init__(self, output_dir: str):
        """
        初始化可视化工具
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def create_debug_visualizations(self, original_img: np.ndarray, colonies: List[Dict]):
        """创建调试可视化"""
        try:
            # 1. 创建检测结果概览
            self.create_detection_overview(original_img, colonies)

            # 2. 创建个体菌落可视化
            self.create_individual_visualizations(colonies)

            # 3. 创建统计图表
            self.create_statistics_plots(colonies)

            logging.info(f"调试可视化已生成: {self.viz_dir}")

        except Exception as e:
            logging.error(f"生成可视化时出错: {e}")

    def create_detection_overview(self, original_img: np.ndarray, colonies: List[Dict]):
        """创建检测结果概览"""
        if not colonies:
            return

        # 创建带标注的图像
        annotated_img = original_img.copy()

        for i, colony in enumerate(colonies):
            # 绘制边界框
            bbox = colony.get('bbox', (0, 0, 0, 0))
            minr, minc, maxr, maxc = bbox
            cv2.rectangle(annotated_img, (minc, minr),
                          (maxc, maxr), (0, 255, 0), 2)

            # 添加标签
            label = colony.get('well_position') or colony.get('id', f'C{i}')
            cv2.putText(annotated_img, label, (minc, minr-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 保存图像
        overview_path = self.viz_dir / 'detection_overview.jpg'
        cv2.imwrite(str(overview_path), cv2.cvtColor(
            annotated_img, cv2.COLOR_RGB2BGR))

    def create_individual_visualizations(self, colonies: List[Dict]):
        """为每个菌落创建个体可视化"""
        individual_dir = self.viz_dir / 'individual_colonies'
        individual_dir.mkdir(exist_ok=True)

        for colony in colonies:
            if 'img' not in colony or 'mask' not in colony:
                continue

            colony_id = colony.get(
                'well_position') or colony.get('id', 'unknown')

            # 创建多面板图像
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 原始图像
            axes[0].imshow(colony['img'])
            axes[0].set_title('Original')
            axes[0].axis('off')

            # 掩码
            axes[1].imshow(colony['mask'], cmap='gray')
            axes[1].set_title('Mask')
            axes[1].axis('off')

            # 叠加图像
            overlay = colony['img'].copy()
            overlay[colony['mask'] > 0] = overlay[colony['mask']
                                                  > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title('Overlay')
            axes[2].axis('off')

            plt.suptitle(f'Colony {colony_id}')
            plt.tight_layout()

            # 保存
            fig_path = individual_dir / f'{colony_id}_analysis.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()

    def create_statistics_plots(self, colonies: List[Dict]):
        """创建统计图表"""
        if not colonies:
            return

        # 收集统计数据
        areas = [colony.get('area', 0) for colony in colonies]
        sam_scores = [colony.get('sam_score', 0)
                      for colony in colonies if 'sam_score' in colony]

        # 创建统计图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 面积分布
        axes[0, 0].hist(areas, bins=20, alpha=0.7)
        axes[0, 0].set_title('Area Distribution')
        axes[0, 0].set_xlabel('Area (pixels)')
        axes[0, 0].set_ylabel('Count')

        # SAM分数分布
        if sam_scores:
            axes[0, 1].hist(sam_scores, bins=20, alpha=0.7)
            axes[0, 1].set_title('SAM Score Distribution')
            axes[0, 1].set_xlabel('SAM Score')
            axes[0, 1].set_ylabel('Count')

        # 面积vs分数散点图
        if sam_scores and len(sam_scores) == len(areas):
            axes[1, 0].scatter(areas, sam_scores, alpha=0.6)
            axes[1, 0].set_title('Area vs SAM Score')
            axes[1, 0].set_xlabel('Area (pixels)')
            axes[1, 0].set_ylabel('SAM Score')

        # 检测方法分布
        methods = [colony.get('detection_method', 'unknown')
                   for colony in colonies]
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1

        if method_counts:
            axes[1, 1].bar(method_counts.keys(), method_counts.values())
            axes[1, 1].set_title('Detection Method Distribution')
            axes[1, 1].set_ylabel('Count')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # 保存
        stats_path = self.viz_dir / 'statistics.png'
        plt.savefig(stats_path, dpi=150, bbox_inches='tight')
        plt.close()


# colony_analysis/utils/__init__.py

__all__ = [
    'LogManager',
    'ResultManager',
    'Visualizer',
    'ImageValidator',
    'DataValidator'
]
