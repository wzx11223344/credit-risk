# -*- coding: utf-8 -*-
"""
信用风险建模工具包 (Credit Risk Modeling Toolkit)
===================================================

专业级信用风险评估系统，包含以下核心模块：

    data      - 财务数据获取（基于akshare，A股上市公司真实数据）
    kmv_model - KMV-Merton结构化违约模型
    copula    - Gaussian/Student-t/Clayton Copula依赖结构建模
    var       - 信用VaR、预期损失、经济资本计算
    scoring   - Altman Z-Score信用评分模型
    report    - HTML报告与可视化图表生成
"""

__version__ = "1.0.0"
__author__ = "Credit Risk Team"
__license__ = "MIT"

# 子模块导入（延迟加载以避免不必要的依赖）
from . import data
from . import kmv_model
from . import copula
from . import var
from . import scoring
from . import report

__all__ = [
    "data",
    "kmv_model",
    "copula",
    "var",
    "scoring",
    "report",
]
