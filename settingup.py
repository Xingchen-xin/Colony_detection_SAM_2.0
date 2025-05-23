#!/usr/bin/env python3
"""
一键生成Colony_detection_SAM_2.0项目结构脚本
运行此脚本将创建完整的项目目录结构和必要文件
"""

import os
from pathlib import Path


# 项目目录结构定义
PROJECT_STRUCTURE = {
    'colony_analysis': {
        '__init__.py': '''"""
Colony Analysis Package - 基于SAM的链霉菌菌落检测和分析工具

版本: 2.0
作者: Colony Analysis Team
"""

__version__ = "2.0.0"
__author__ = "Colony Analysis Team"

from .config import ConfigManager
from .core import SAMModel, ColonyDetector
from .analysis import ColonyAnalyzer
from .utils import LogManager, ResultManager, Visualizer

__all__ = [
    'ConfigManager',
    'SAMModel', 
    'ColonyDetector',
    'ColonyAnalyzer',
    'LogManager',
    'ResultManager', 
    'Visualizer'
]
''',
        'config': {
            '__init__.py': '''from .settings import ConfigManager

__all__ = ['ConfigManager']
''',
            'settings.py': '# 配置管理器代码将放置在这里'
        },
        'core': {
            '__init__.py': '''from .sam_model import SAMModel
from .detection import ColonyDetector

__all__ = ['SAMModel', 'ColonyDetector']
''',
            'sam_model.py': '# 统一的SAM模型封装代码将放置在这里',
            'detection.py': '# 菌落检测器代码将放置在这里'
        },
        'analysis': {
            '__init__.py': '''from .colony import ColonyAnalyzer
from .features import FeatureExtractor
from .scoring import ScoringSystem

__all__ = ['ColonyAnalyzer', 'FeatureExtractor', 'ScoringSystem']
''',
            'colony.py': '# 菌落分析器代码将放置在这里',
            'features.py': '# 特征提取代码将放置在这里',
            'scoring.py': '# 评分系统代码将放置在这里'
        },
        'utils': {
            '__init__.py': '''from .logging import LogManager
from .results import ResultManager
from .visualization import Visualizer
from .validation import ImageValidator, DataValidator

__all__ = [
    'LogManager',
    'ResultManager', 
    'Visualizer',
    'ImageValidator',
    'DataValidator'
]
''',
            'logging.py': '# 日志管理代码将放置在这里',
            'results.py': '# 结果管理代码将放置在这里',
            'visualization.py': '# 可视化工具代码将放置在这里',
            'validation.py': '# 数据验证代码将放置在这里'
        }
    },
    'tests': {
        '__init__.py': '',
        'test_sam_model.py': '# SAM模型测试',
        'test_detection.py': '# 检测功能测试',
        'test_analysis.py': '# 分析功能测试',
        'test_config.py': '# 配置管理测试'
    },
    'models': {
        '.gitkeep': '# 模型文件存放目录，SAM权重文件将放置在这里'
    },
    'examples': {
        'images': {
            '.gitkeep': '# 示例图像存放目录'
        },
        'notebooks': {
            '.gitkeep': '# Jupyter notebook示例'
        }
    },
    'docs': {
        '.gitkeep': '# 文档目录'
    }
}

# 根目录文件内容
ROOT_FILES = {
    'main.py': '''#!/usr/bin/env python3
"""
Colony Detection SAM 2.0 - 主入口文件
基于SAM的链霉菌菌落检测和分析工具
"""

import os
import time
import argparse
import cv2
import logging

# TODO: 替换为实际的导入
# from colony_analysis.config import ConfigManager
# from colony_analysis.utils import LogManager
# from colony_analysis.core import SAMModel, ColonyDetector
# from colony_analysis.analysis import ColonyAnalyzer
# from colony_analysis.utils import ResultManager, Visualizer


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
    # TODO: 实现环境设置
    print("正在设置环境...")
    pass


def load_and_validate_image(image_path):
    """加载和验证图像"""
    # TODO: 实现图像加载和验证
    print(f"正在加载图像: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def main():
    """主函数 - 保持简洁，主要负责流程协调"""
    try:
        print("Colony Detection SAM 2.0 - 链霉菌菌落分析工具")
        print("=" * 50)
        
        # 1. 解析参数和初始化
        args = parse_arguments()
        config_manager, log_manager = setup_environment(args)
        
        print(f"开始分析图像: {args.image}")
        start_time = time.time()
        
        # 2. 加载图像
        img_rgb = load_and_validate_image(args.image)
        print(f"图像尺寸: {img_rgb.shape}")
        
        # TODO: 实现以下步骤
        print("TODO: 以下功能待实现")
        print("3. 初始化SAM模型")
        print("4. 执行菌落检测")
        print("5. 执行特征分析")
        print("6. 保存结果")
        print("7. 生成可视化")
        
        # 计算运行时间
        elapsed_time = time.time() - start_time
        print(f"处理完成，耗时 {elapsed_time:.2f} 秒")
        
    except Exception as e:
        print(f"程序执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
''',

    'config.yaml': '''# Colony Detection SAM 2.0 配置文件

detection:
  model_type: vit_b              # SAM模型类型: vit_b, vit_l, vit_h
  mode: auto                     # 检测模式: auto, grid, hybrid
  min_colony_area: 5000          # 最小菌落面积
  expand_pixels: 8               # 掩码扩展像素数
  merge_overlapping: true        # 是否合并重叠菌落
  use_preprocessing: true        # 是否使用图像预处理
  overlap_threshold: 0.3         # 重叠阈值

sam:
  points_per_side: 64            # 每边采样点数
  pred_iou_thresh: 0.85          # IoU阈值
  stability_score_thresh: 0.8    # 稳定性分数阈值
  min_mask_region_area: 1500     # 最小掩码区域面积
  crop_n_layers: 1               # 裁剪层数
  crop_n_points_downscale_factor: 1  # 下采样因子

analysis:
  advanced: false                # 是否启用高级分析
  learning_enabled: false        # 是否启用学习系统
  aerial_threshold: 0.6          # 气生菌丝阈值
  metabolite_threshold: 0.5      # 代谢产物阈值
  enable_parallel: false         # 是否启用并行处理
  max_workers: 4                 # 最大工作线程数

output:
  debug: false                   # 是否生成调试输出
  well_plate: false              # 是否使用孔板编号
  rows: 8                        # 孔板行数
  cols: 12                       # 孔板列数
  save_masks: true               # 是否保存掩码
  save_visualizations: true      # 是否保存可视化
  image_format: jpg              # 图像输出格式

logging:
  level: INFO                    # 日志级别: DEBUG, INFO, WARNING, ERROR
  log_to_file: true              # 是否记录到文件
  log_dir: null                  # 日志目录 (null = 自动)
  max_log_files: 10              # 最大日志文件数
''',

    'requirements.txt': '''# Colony Detection SAM 2.0 依赖包

# 核心依赖
torch>=1.7.1
torchvision>=0.8.2
numpy>=1.19.2
opencv-python>=4.5.1
pillow>=8.0.0

# SAM依赖
segment-anything
git+https://github.com/facebookresearch/segment-anything.git

# 图像处理
scikit-image>=0.18.1
scipy>=1.6.0

# 数据处理和分析
pandas>=1.2.0
scikit-learn>=0.24.1

# 可视化
matplotlib>=3.3.3
seaborn>=0.11.1

# 配置文件处理
pyyaml>=5.4.1

# 进度显示
tqdm>=4.56.0

# 并行处理
joblib>=1.0.0

# 测试框架
pytest>=6.0.0
pytest-cov>=2.10.0

# 代码质量
black>=21.0.0
flake8>=3.8.0
isort>=5.7.0

# 类型检查
mypy>=0.800

# 文档生成
sphinx>=3.4.0
sphinx-rtd-theme>=0.5.0
''',

    'README.md': '''# Colony Detection SAM 2.0

基于Segment Anything Model (SAM)的链霉菌菌落检测和分析工具

## 功能特点

- 🔬 高精度菌落分割和检测
- 📊 全面的形态学特征分析  
- 🎨 代谢产物识别和定量
- 📈 智能评分和表型分类
- 🔧 支持96孔板自动识别
- 📋 丰富的输出格式和可视化

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载SAM模型

下载相应的SAM模型权重文件到 `models/` 目录：

- [vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- [vit_l](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)  
- [vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

### 3. 基本使用

```bash
# 基本分析
python main.py --image your_image.jpg --output results/

# 高级分析模式
python main.py --image your_image.jpg --output results/ --advanced --debug

# 96孔板模式
python main.py --image plate.jpg --well-plate --mode grid
```

## 项目结构

```
Colony_detection_SAM_2.0/
├── main.py                    # 主入口文件
├── config.yaml               # 配置文件
├── colony_analysis/          # 主包
│   ├── config/              # 配置管理
│   ├── core/                # 核心功能
│   ├── analysis/            # 分析模块
│   └── utils/               # 工具模块
├── tests/                   # 测试文件
├── models/                  # 模型权重存放目录
└── examples/               # 示例和文档
```

## 配置说明

主要配置参数在 `config.yaml` 中：

- `detection`: 检测相关参数
- `sam`: SAM模型参数
- `analysis`: 分析功能参数
- `output`: 输出格式参数

## 输出说明

程序会在输出目录生成：

- `results/analysis_results.csv`: 分析结果表格
- `colonies/`: 单个菌落图像
- `visualizations/`: 检测和分析可视化
- `reports/`: 分析报告

## 开发指南

### 环境设置

```bash
# 安装开发依赖
pip install -r requirements.txt

# 代码格式化
black .
isort .

# 运行测试
pytest tests/
```

## 版本历史

- **v2.0**: 架构重构，模块化设计
- **v1.0**: 初始版本

## 许可证

Apache 2.0 License

## 贡献指南

欢迎提交Issue和Pull Request！
''',

    'setup.py': '''from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="colony-detection-sam",
    version="2.0.0",
    author="Colony Analysis Team",
    author_email="",
    description="基于SAM的链霉菌菌落检测和分析工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "colony-detect=main:main",
        ],
    },
)
''',

    '.gitignore': '''# Colony Detection SAM 2.0 - Git忽略文件

# Python缓存文件
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# 虚拟环境
venv/
env/
ENV/
.venv/
.env/
.ENV/
conda-env/

# IDE和编辑器
.vscode/
.idea/
*.sublime-project
*.sublime-workspace
.spyderproject
.spyproject
.ropeproject
*.swp
*.swo
*~

# 操作系统文件
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
desktop.ini

# 日志文件
*.log
logs/
*.out

# 模型文件（通常很大，需要单独管理）
models/*.pth
models/*.onnx
models/*.pb
models/*.pt
models/*.h5
models/*.pkl
!models/.gitkeep

# 输出目录
output/
results/
debug/
temp/
tmp/

# 数据文件
data/
*.csv
*.xlsx
*.json
!config.yaml
!requirements.txt

# 图像文件（示例图像除外）
*.jpg
*.jpeg
*.png
*.tif
*.tiff
*.bmp
!examples/images/.gitkeep

# 测试覆盖率报告
htmlcov/
.coverage
.coverage.*
.cache
.pytest_cache/
coverage.xml
*.cover
.hypothesis/

# 文档生成
_build/
.doctrees/
docs/_build/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# 配置文件备份
config.yaml.bak
config.json.bak
settings.ini.bak

# 临时文件
*.tmp
*.temp
.temporary/

# 压缩文件
*.zip
*.tar.gz
*.rar
*.7z

# 性能分析文件
*.prof
*.lprof

# mypy类型检查
.mypy_cache/
.dmypy.json
dmypy.json

# 数据库文件
*.db
*.sqlite
*.sqlite3

# 环境变量文件
.env.local
.env.development
.env.test
.env.production
'''
}


def create_directory_structure(base_path, structure, current_path=""):
    """递归创建目录结构"""
    for name, content in structure.items():
        full_path = os.path.join(base_path, current_path, name)

        if isinstance(content, dict):
            # 这是一个目录
            os.makedirs(full_path, exist_ok=True)
            print(f"📁 创建目录: {full_path}")

            # 递归创建子结构
            create_directory_structure(
                base_path, content, os.path.join(current_path, name))
        else:
            # 这是一个文件
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"📄 创建文件: {full_path}")


def create_root_files(base_path, files):
    """创建根目录文件"""
    for filename, content in files.items():
        file_path = os.path.join(base_path, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📄 创建根文件: {file_path}")


def main():
    """主函数"""
    print("🚀 Colony Detection SAM 2.0 项目结构生成器")
    print("=" * 50)

    # 获取当前目录
    base_path = os.getcwd()
    print(f"📍 当前目录: {base_path}")

    # 检查是否在正确的目录
    if not base_path.endswith('Colony_detection_SAM_2.0'):
        print("⚠️  警告: 当前目录名不是 'Colony_detection_SAM_2.0'")
        confirm = input("是否继续在当前目录创建项目结构? (y/N): ")
        if confirm.lower() != 'y':
            print("❌ 已取消操作")
            return

    print("\n🏗️  开始创建项目结构...")

    try:
        # 创建目录结构
        create_directory_structure(base_path, PROJECT_STRUCTURE)

        print("\n📋 创建根目录文件...")
        # 创建根目录文件
        create_root_files(base_path, ROOT_FILES)

        print("\n✅ 项目结构创建完成!")
        print("\n📋 项目结构概览:")
        print(f"""
Colony_detection_SAM_2.0/
├── main.py                    # ✅ 主入口文件
├── config.yaml               # ✅ 配置文件
├── requirements.txt          # ✅ 依赖列表
├── README.md                 # ✅ 项目说明
├── setup.py                  # ✅ 安装脚本
├── .gitignore                # ✅ Git忽略文件
├── colony_analysis/          # ✅ 主包目录
│   ├── config/              # ✅ 配置管理
│   ├── core/                # ✅ 核心功能
│   ├── analysis/            # ✅ 分析模块
│   └── utils/               # ✅ 工具模块
├── tests/                   # ✅ 测试目录
├── models/                  # ✅ 模型存放目录
├── examples/                # ✅ 示例目录
└── docs/                    # ✅ 文档目录
        """)

        print("\n🎯 下一步操作:")
        print("1. 将重构后的代码文件复制到对应目录")
        print("2. 下载SAM模型权重到 models/ 目录")
        print("3. 安装依赖: pip install -r requirements.txt")
        print("4. 运行测试: python main.py --help")

        print("\n📁 代码文件复制指南:")
        print("- SAMModel类代码 → colony_analysis/core/sam_model.py")
        print("- ColonyDetector类代码 → colony_analysis/core/detection.py")
        print("- ConfigManager类代码 → colony_analysis/config/settings.py")
        print("- ResultManager类代码 → colony_analysis/utils/results.py")
        print("- Visualizer类代码 → colony_analysis/utils/visualization.py")
        print("- 其他工具类代码 → colony_analysis/utils/ 对应文件")

    except Exception as e:
        print(f"❌ 创建项目结构时发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
