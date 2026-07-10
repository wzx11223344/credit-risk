# 信用风险建模工具 (Credit Risk Modeling Toolkit)

专业级信用风险评估系统，使用真实A股财务数据实现KMV-Merton结构化违约模型、Copula依赖结构、信用VaR与经济资本计算、Altman Z-Score评分。

## 功能特性

### KMV-Merton 结构化违约模型
- 将股权视为公司资产的看涨期权，通过Black-Scholes方程迭代求解资产价值和波动率
- 计算违约距离(Distance to Default, DD)和预期违约频率(EDF)
- 支持牛顿迭代/固定点迭代，保证收敛性

### Copula 依赖结构建模
- Gaussian Copula（全局相依结构）
- Student-t Copula（厚尾相依捕捉）
- Clayton Copula（下尾相依，适合信用风险）
- Kendall秩相关系数估计，最大似然参数拟合

### 信用VaR与经济资本
- 预期损失(EL) = PD x LGD x EAD
- 意外损失(UL)，组合UL考虑违约相关性
- 信用VaR（解析法 + 蒙特卡洛模拟）
- 经济资本(EC)计算，边际风险贡献分解

### Altman Z-Score 信用评分
- 经典5因子模型（营运资金/总资产、留存收益/总资产、EBIT/总资产、市值/负债、营收/总资产）
- 区域判定（安全区/灰色区/危险区）
- 信用等级映射（AAA ~ CCC）
- 多维财务比率综合评分

### 可视化HTML报告
- 违约距离排名图、资产价值vs违约点图
- 信用评分分布图、Copula热力图
- 组合损失分布直方图
- 信用风险预警表（高/中/低风险标记）

## 快速开始

### 安装

```bash
cd credit-risk
pip install -r requirements.txt
```

依赖包：
- akshare >= 1.12.0（A股数据获取）
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- rich >= 13.0.0（CLI进度显示）

### 命令行使用

```bash
# 评估指定股票（完整分析，含HTML报告）
python assess.py --tickers 600519,000858,601318 --method all

# 仅KMV模型分析
python assess.py --tickers 600519,000858 --method kmv

# 仅信用评分
python assess.py --tickers 600519,000858 --method scoring

# Copula依赖结构分析
python assess.py --tickers 600519,000858,601318 --method copula

# 信用VaR与经济资本
python assess.py --tickers 600519,000858,601318 --method var

# 使用指数成分股（沪深300前10大）
python assess.py --index hs300 --top 10 --method all

# 自定义无风险利率和模拟次数
python assess.py --tickers 600519,000858 --method all --rf 0.03 --sim 50000
```

### Python API

```python
from credit_risk import data, kmv_model, scoring, copula as copula_mod, var as var_mod

# 1. 获取财务数据
panel = data.build_credit_panel("600519")
print(f"市值: {panel['market_cap']/1e8:.2f}亿")
print(f"违约点: {panel['default_point']/1e8:.2f}亿")

# 2. KMV模型评估
kmv = kmv_model.kmv_assess(panel)
print(f"资产价值: {kmv['asset_value']/1e8:.2f}亿")
print(f"资产波动率: {kmv['asset_volatility']:.4f}")
print(f"违约距离DD: {kmv['dd']:.3f}")
print(f"违约概率EDF: {kmv['edf']:.4%}")

# 3. 信用评分
score = scoring.scoring_assess(panel)
print(f"Altman Z-Score: {score['altman_z']:.3f}")
print(f"区域: {score['altman_zone']}")
print(f"评级: {score['rating']}")

# 4. 多公司分析
panels = data.build_multi_company_panel(["600519", "000858", "601318"])
kmv_df = kmv_model.kmv_assess_multi(panels)

# 5. Copula分析
copula_result = copula_mod.portfolio_default_correlation(panels, kmv_df, "gaussian")

# 6. 组合信用VaR
var_result = var_mod.portfolio_credit_assessment(kmv_df, copula_result)
print(f"信用VaR: {var_result['portfolio_summary']['credit_var']/1e8:.4f}亿")
```

## 项目结构

```
credit-risk/
├── assess.py                  # CLI入口
├── credit_risk/
│   ├── __init__.py            # 包初始化
│   ├── data.py               # 财务数据获取（akshare）
│   ├── kmv_model.py          # KMV-Merton结构化模型
│   ├── copula.py             # Copula依赖结构建模
│   ├── var.py                # 信用VaR与预期损失
│   ├── scoring.py            # 信用评分模型
│   └── report.py             # HTML报告生成
├── SKILL.md                  # 技能说明文档
├── README.md                 # 本文件
├── requirements.txt          # 依赖列表
└── output/                   # 输出目录（自动创建）
    ├── charts/               # 图表PNG
    ├── kmv_results.csv       # KMV结果
    ├── scoring_results.csv   # 评分结果
    ├── copula_corr_matrix.csv # 相关矩阵
    ├── risk_contributions.csv # 风险贡献
    └── credit_risk_report_*.html # HTML报告
```

## 核心模型说明

### KMV-Merton 模型

股权视为公司资产的看涨期权：

```
E = V·N(d1) - D·e^(-rT)·N(d2)      (Black-Scholes)
σ_E·E = N(d1)·σ_A·V                (Ito引理)

d1 = [ln(V/D) + (r + 0.5σ_A²)T] / (σ_A·√T)
d2 = d1 - σ_A·√T

DD = [ln(V/D) + (μ - 0.5σ_A²)T] / (σ_A·√T)
EDF = N(-DD)
```

迭代求解：初始化 σ_A，用Brent法求解V，更新 σ_A，直至收敛。

### Altman Z-Score

```
Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5

X1 = 营运资金 / 总资产
X2 = 留存收益 / 总资产
X3 = EBIT / 总资产
X4 = 股权市值 / 总负债
X5 = 营业收入 / 总资产

Z > 2.99  : 安全区
1.81~2.99 : 灰色区
Z < 1.81  : 危险区
```

### 信用VaR

```
EL = PD × LGD × EAD
UL = EAD × LGD × √(PD × (1-PD))
UL_portfolio = √(ΣΣ ρ_ij × UL_i × UL_j)
Credit VaR = Z_α × UL_portfolio
经济资本 EC = Credit VaR
```

## 数据来源

所有财务数据来自akshare公开接口：
- 资产负债表：`stock_balance_sheet_by_report_em`
- 利润表：`stock_profit_sheet_by_report_em`
- 个股信息：`stock_individual_info_em`
- 历史行情：`stock_zh_a_hist`（前复权）
- 国债收益率：`bond_china_yield`

不使用任何随机数或伪造数据。蒙特卡洛模拟使用固定随机种子(random_state=42)保证可复现。

## 许可证

MIT License
