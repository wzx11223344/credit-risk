---
slug: credit-risk
displayName: 信用风险建模工具
version: 1.0.0
summary: 专业级信用风险评估系统，实现KMV-Merton结构化违约模型、Gaussian/t/Clayton Copula依赖结构、信用VaR与经济资本计算、Altman Z-Score评分，使用真实A股财务数据。
tags:
  - finance
  - credit-risk
  - kmv-model
  - copula
  - quantitative-finance
license: MIT
---

# 信用风险建模工具

## 能力说明

本工具实现了一套完整的信用风险建模与分析流程，涵盖从数据获取到风险度量的全链路：

| 模块 | 功能 | 数学模型 |
|------|------|----------|
| `data.py` | A股财务数据获取 | akshare接口（资产负债表、利润表、行情数据） |
| `kmv_model.py` | 结构化违约模型 | KMV-Merton模型、Black-Scholes期权定价、牛顿迭代 |
| `copula.py` | 违约依赖结构 | Gaussian/t/Clayton Copula、Kendall's tau、MLE |
| `var.py` | 信用风险度量 | Credit VaR、EL/UL、经济资本、蒙特卡洛模拟 |
| `scoring.py` | 信用评分 | Altman Z-Score 5因子模型、综合评分 |
| `report.py` | 可视化报告 | matplotlib图表 + base64嵌入HTML |

## 能力边界说明

1. **数据来源**：仅支持A股上市公司（通过akshare获取），暂不支持港股/美股/债券
2. **KMV模型假设**：基于Merton结构化模型假设，假设违约仅在到期时发生、资产价值服从几何布朗运动
3. **Copula限制**：Clayton Copula仅支持双变量；Student-t Copula的多元CDF使用蒙特卡洛近似
4. **VaR方法**：提供解析法（正态近似）和蒙特卡洛法两种，非完整的CreditMetrics体系
5. **Altman Z-Score**：使用原始公开公司版公式，对中国A股的适配性可能有限（X5=营收/总资产对非制造业可能失真）
6. **无风险利率**：默认使用中国1年期国债收益率，获取失败时使用2.5%默认值
7. **市场数据**：股价波动率基于最近252个交易日对数收益率年化标准差

## FAQ

### Q1: KMV模型迭代不收敛怎么办？
迭代使用固定点法+ Brent求根。若不收敛，通常原因是：(a) 股权价值极小（接近资不抵债）；(b) 股价波动率异常高（>200%）；(c) 违约点为零。代码已设置最大迭代500次和多种兜底逻辑。若仍不收敛，结果中 `converged=False`，此时结果仅供参考。

### Q2: 为什么EDF（违约概率）和实际违约率差异大？
KMV模型计算的EDF是理论违约概率（风险中性或真实概率），基于Merton模型的强假设。实际违约还受流动性危机、宏观冲击、治理风险等非模型因素影响。EDF更适合作为相对风险排序工具，而非绝对违约预测。

### Q3: Copula分析需要多少家公司？
至少2家。Gaussian和Student-t Copula支持任意维度；Clayton Copula仅支持双变量（取前两家）。建议3-10家公司以获得有意义的相关性估计。

### Q4: 蒙特卡洛模拟的随机种子为什么固定为42？
为保证结果可复现。所有涉及随机数生成的场景（蒙特卡洛模拟、Copula采样）均设置 `random_state=42`，不使用 `np.random` 生成业务数据。改变种子会改变损失分布的具体形态，但统计特征（均值、分位数）保持稳定。

### Q5: Altman Z-Score对中国A股适用吗？
原始Z-Score系数基于1968年美国制造业上市公司数据。对中国A股，建议同时参考：(a) Z''-Score非制造业版（去掉X5）；(b) 综合评分（多维加权）。工具默认使用公开公司版，用户可通过API调用切换变体。

### Q6: 如何自定义违约风险暴露（EAD）和违约损失率（LGD）？
`portfolio_credit_assessment()` 函数接受 `eads` 和 `lgds` 参数。默认EAD为每家公司1亿元等权重，LGD为0.55（恢复率45%）。用户可传入实际敞口和回收率数据。

### Q7: akshare接口报错或数据为空怎么办？
可能原因：(a) 网络连接问题；(b) akshare版本过低（需>=1.12.0）；(c) 股票代码格式错误（应为6位数字如600519）；(d) 新上市公司财报数据不全。建议先 `pip install --upgrade akshare` 再重试。

## 输出示例

### KMV模型输出
```
代码      名称      资产价值(亿)  违约点(亿)  DD      EDF(%)    收敛
600519   贵州茅台   21500.32     820.15     5.832   0.0003   是
000858   五粮液    6800.45      350.22     4.921   0.0004   是
601318   中国平安   12800.78     9800.56    1.245   0.1064   是
```

### 信用评分输出
```
代码      名称      Z-Score  区域     评级   综合评分  评级
600519   贵州茅台   8.521   安全区    AAA   92.3     AAA
000858   五粮液    6.332   安全区    AA    85.7     AA
601318   中国平安   2.145   灰色区    B     52.1     BB
```

### 组合VaR输出
```
总风险暴露: 3.00 亿
预期损失EL: 0.0584 亿
意外损失UL: 0.3125 亿
信用VaR(99.9%): 0.9173 亿
经济资本EC: 0.9173 亿
蒙特卡洛VaR(99.9%): 0.8932 亿
最坏损失: 1.65 亿
```

## 安装与使用

### 安装依赖

```bash
cd credit-risk
pip install -r requirements.txt
```

### 命令行使用

```bash
# 评估指定股票（完整分析）
python assess.py --tickers 600519,000858,601318 --method all

# 仅KMV模型分析
python assess.py --tickers 600519,000858 --method kmv

# 使用沪深300前10大成分股
python assess.py --index hs300 --top 10 --method all

# 指定无风险利率和模拟次数
python assess.py --tickers 600519,000858 --method var --rf 0.03 --sim 50000
```

### Python API调用

```python
from credit_risk import data, kmv_model, scoring

# 获取财务数据
panel = data.build_credit_panel("600519")

# KMV模型评估
kmv = kmv_model.kmv_assess(panel)
print(f"DD={kmv['dd']:.3f}, EDF={kmv['edf']:.4%}")

# 信用评分
score = scoring.scoring_assess(panel)
print(f"Z-Score={score['altman_z']:.3f}, 评级={score['rating']}")
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `output/kmv_results.csv` | KMV模型结果 |
| `output/scoring_results.csv` | 信用评分结果 |
| `output/copula_corr_matrix.csv` | Copula相关矩阵 |
| `output/risk_contributions.csv` | 边际风险贡献 |
| `output/credit_risk_report_*.html` | HTML完整报告 |
| `output/charts/*.png` | 可视化图表 |
