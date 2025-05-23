#!/usr/bin/env python3
"""
Colony Detection SAM 2.0 - 主入口文件
基于SAM的链霉菌菌落检测和分析工具
"""

import os
import time
import argparse
import cv2
import logging

from colony_analysis.config import ConfigManager
from colony_analysis.utils import LogManager
from colony_analysis.core import SAMModel, ColonyDetector
from colony_analysis.analysis import ColonyAnalyzer
from colony_analysis.utils import ResultManager, Visualizer


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="基于SAM的链霉菌菌落分析工具 v2.0")
    
    # 基本参数
    parser.add_argument('--image', '-i', required=True, help='输入图像路径')
    parser.add_argument('--output', '-o', default='output', help='输出目录')
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    # 检测参数
    parser.add_argument('--mode', '-m', choices=['auto', 'grid', 'hybrid'], 
                        default='auto', help='检测模式')
    parser.add_argument('--model', choices=['vit_b', 'vit_l', 'vit_h'], 
                        default='vit_b', help='SAM模型类型')
    
    # 分析参数
    parser.add_argument('--advanced', '-a', action='store_true', 
                        help='启用高级特征分析')
    parser.add_argument('--debug', '-d', action='store_true', 
                        help='生成调试图像')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='输出详细日志')
    
    # 孔板参数
    parser.add_argument('--well-plate', action='store_true',
                        help='使用96孔板编号系统(A1-H12)')
    parser.add_argument('--rows', type=int, default=8, help='孔板行数')
    parser.add_argument('--cols', type=int, default=12, help='孔板列数')
    
    return parser.parse_args()


def setup_environment(args):
    """设置环境和配置"""
    # 初始化配置管理器
    config_manager = ConfigManager(args.config)
    config_manager.update_from_args(args)

    # 设置日志系统
    log_manager = LogManager(config_manager)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    return config_manager, log_manager


def load_and_validate_image(image_path):
    """加载和验证图像"""
    from colony_analysis.utils.validation import ImageValidator

    # 加载图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 验证图像
    is_valid, error_msg = ImageValidator.validate_image(img_rgb)
    if not is_valid:
        raise ValueError(f"图像验证失败: {error_msg}")

    return img_rgb


def main():
    """主函数 - 保持简洁，主要负责流程协调"""
    try:
        # 1. 解析参数和初始化
        args = parse_arguments()
        config_manager, log_manager = setup_environment(args)

        logging.info(f"开始分析图像: {args.image}")
        start_time = time.time()

        # 2. 加载图像
        img_rgb = load_and_validate_image(args.image)
        logging.info(f"图像尺寸: {img_rgb.shape}")

        # 3. 初始化核心组件
        sam_model = SAMModel(
            model_type=args.model,
            config=config_manager
        )

        detector = ColonyDetector(
            sam_model=sam_model,
            config=config_manager
        )

        analyzer = ColonyAnalyzer(
            sam_model=sam_model,
            config=config_manager
        )

        # 4. 执行检测
        logging.info("开始菌落检测...")
        colonies = detector.detect(img_rgb, mode=args.mode)

        if not colonies:
            logging.warning("未检测到菌落")
            return

        logging.info(f"检测到 {len(colonies)} 个菌落")

        # 5. 执行分析
        logging.info("开始菌落分析...")
        analyzed_colonies = analyzer.analyze(colonies, advanced=args.advanced)

        # 6. 保存结果
        logging.info("保存分析结果...")
        result_manager = ResultManager(args.output)
        result_manager.save_all_results(analyzed_colonies, args)

        # 7. 生成可视化
        if args.debug:
            visualizer = Visualizer(args.output)
            visualizer.create_debug_visualizations(img_rgb, analyzed_colonies)

        # 8. 显示总结
        elapsed_time = time.time() - start_time
        logging.info(f"分析完成，耗时 {elapsed_time:.2f} 秒")
        logging.info(f"结果已保存到: {args.output}")

    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
