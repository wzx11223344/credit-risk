# -*- coding: utf-8 -*-
"""
信用评分模型模块
===================

本模块实现基于财务指标的信用评分，核心包括：
    - Altman Z-Score模型（经典5因子模型）
    - Z-Score变体（Z'-Score私有公司版, Z''-Score非制造业版）
    - Z-Score解读（安全区/灰色区/危险区）
    - 财务比率综合评分
    - 评分等级映射（AAA/AA/A/BBB/BB/B/CCC）
    - 行业基准比较

Altman Z-Score 公式（原始公开制造业版）：
    Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5

    X1 = 营运资金 / 总资产 = (流动资产 - 流动负债) / 总资产
    X2 = 留存收益 / 总资产
    X3 = EBIT / 总资产 = (利润总额 + 财务费用) / 总资产
    X4 = 股权市值 / 总负债账面价值
    X5 = 营业收入 / 总资产
"""

import numpy as np
import pandas as pd


# ============================================================
#  Altman Z-Score 计算
# ============================================================

def compute_altman_zscore(
    current_assets: float,
    current_liab: float,
    total_assets: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_liab: float,
    revenue: float,
    variant: str = "public",
) -> dict:
    """
    计算Altman Z-Score及其5个因子。

    支持三种变体：
        - "public"      : 原始公开公司版（制造业）
        - "private"     : 私有公司版（Z'-Score）
        - "non_mfg"     : 非制造业版（Z''-Score）

    参数:
        current_assets  : 流动资产
        current_liab    : 流动负债
        total_assets    : 总资产
        retained_earnings: 留存收益
        ebit            : 息税前利润 (EBIT = 利润总额 + 财务费用)
        market_cap      : 股权市值
        total_liab      : 总负债（账面价值）
        revenue         : 营业收入
        variant         : 模型变体 "public"/"private"/"non_mfg"

    返回:
        dict，包含：
            - Z            : Z-Score总分
            - X1~X5        : 5个因子值
            - variant      : 使用的模型变体
            - zone         : 区域判定（安全区/灰色区/危险区）
            - zone_detail  : 区域详细描述
    """
    # 计算各因子
    working_capital = current_assets - current_liab
    X1 = working_capital / total_assets if total_assets > 0 else 0.0
    X2 = retained_earnings / total_assets if total_assets > 0 else 0.0
    X3 = ebit / total_assets if total_assets > 0 else 0.0

    if variant == "public":
        # X4 = 股权市值 / 总负债
        X4 = market_cap / total_liab if total_liab > 0 else 0.0
        # X5 = 营业收入 / 总资产
        X5 = revenue / total_assets if total_assets > 0 else 0.0
        Z = 1.2 * X1 + 1.4 * X2 + 3.3 * X3 + 0.6 * X4 + 1.0 * X5
    elif variant == "private":
        # Z'-Score: X4 用账面价值
        # X4 = 股东权益账面值 / 总负债
        # 这里用 market_cap 近似（因为公开公司）
        X4 = market_cap / total_liab if total_liab > 0 else 0.0
        X5 = revenue / total_assets if total_assets > 0 else 0.0
        Z = 0.717 * X1 + 0.847 * X2 + 3.107 * X3 + 0.420 * X4 + 0.998 * X5
    elif variant == "non_mfg":
        # Z''-Score: 去掉X5（非制造业版）
        X4 = market_cap / total_liab if total_liab > 0 else 0.0
        X5 = np.nan  # 不使用
        Z = 6.56 * X1 + 3.26 * X2 + 6.72 * X3 + 1.05 * X4
    else:
        raise ValueError(f"不支持的变体: {variant}")

    # 区域判定
    zone, zone_detail = _interpret_zscore(Z, variant)

    return {
        "Z": Z,
        "X1": X1,
        "X2": X2,
        "X3": X3,
        "X4": X4,
        "X5": X5,
        "variant": variant,
        "zone": zone,
        "zone_detail": zone_detail,
    }


def _interpret_zscore(z: float, variant: str = "public") -> tuple:
    """
    解读Z-Score的区域判定。

    原始公开版阈值：
        Z > 2.99  : 安全区（Safe Zone）
        1.81 ≤ Z ≤ 2.99 : 灰色区（Grey Zone）
        Z < 1.81  : 危险区（Distress Zone）

    非制造业版阈值：
        Z > 2.60  : 安全区
        1.10 ≤ Z ≤ 2.60 : 灰色区
        Z < 1.10  : 危险区

    参数:
        z      : Z-Score值
        variant: 模型变体

    返回:
        (zone, zone_detail) 元组
    """
    if np.isnan(z):
        return "未知", "无法计算Z-Score"

    if variant in ("public", "private"):
        if z > 2.99:
            return "安全区", "财务状况良好，短期违约风险很低"
        elif z >= 1.81:
            return "灰色区", "财务状况存在不确定性，需进一步分析"
        else:
            return "危险区", "财务状况堪忧，违约风险较高"
    elif variant == "non_mfg":
        if z > 2.60:
            return "安全区", "财务状况良好，短期违约风险很低"
        elif z >= 1.10:
            return "灰色区", "财务状况存在不确定性，需进一步分析"
        else:
            return "危险区", "财务状况堪忧，违约风险较高"
    else:
        if z > 2.99:
            return "安全区", "财务状况良好"
        elif z >= 1.81:
            return "灰色区", "需进一步分析"
        else:
            return "危险区", "违约风险较高"


# ============================================================
#  信用等级映射
# ============================================================

def zscore_to_rating(z: float) -> str:
    """
    将Z-Score映射为信用评级。

    映射标准（基于经验阈值）：
        Z >= 5.85  : AAA
        4.95 ~ 5.85: AA
        4.15 ~ 4.95: A
        3.65 ~ 4.15: BBB
        2.85 ~ 3.65: BB
        2.15 ~ 2.85: B
        Z < 2.15   : CCC

    参数:
        z: Z-Score值

    返回:
        信用评级字符串，如 "AAA"
    """
    if np.isnan(z):
        return "NR"  # 未评级

    if z >= 5.85:
        return "AAA"
    elif z >= 4.95:
        return "AA"
    elif z >= 4.15:
        return "A"
    elif z >= 3.65:
        return "BBB"
    elif z >= 2.85:
        return "BB"
    elif z >= 2.15:
        return "B"
    else:
        return "CCC"


def edf_to_rating(edf: float) -> str:
    """
    将违约概率（EDF）映射为信用评级。

    基于S&P评级与违约概率的对应关系。

    参数:
        edf: 违约概率（0~1）

    返回:
        信用评级字符串
    """
    if np.isnan(edf):
        return "NR"

    edf_pct = edf * 100  # 转为百分比

    if edf_pct < 0.02:
        return "AAA"
    elif edf_pct < 0.05:
        return "AA"
    elif edf_pct < 0.10:
        return "A"
    elif edf_pct < 0.20:
        return "BBB"
    elif edf_pct < 1.00:
        return "BB"
    elif edf_pct < 5.00:
        return "B"
    elif edf_pct < 20.00:
        return "CCC"
    else:
        return "CC"


# 评级对应的违约概率经验值（用于参考）
RATING_PD_MAP = {
    "AAA": 0.0002,
    "AA": 0.0005,
    "A": 0.001,
    "BBB": 0.003,
    "BB": 0.015,
    "B": 0.050,
    "CCC": 0.150,
    "CC": 0.300,
    "NR": np.nan,
}


# ============================================================
#  财务比率综合评分
# ============================================================

def compute_comprehensive_score(panel: dict) -> dict:
    """
    计算财务比率综合评分。

    基于多个维度的财务比率，加权汇总为0-100的综合评分。

    评分维度及权重：
        - 盈利能力 (25%): ROA, ROE, 净利率
        - 偿债能力 (25%): 资产负债率, 利息保障倍数
        - 流动性 (20%): 流动比率, 速动比率
        - 运营效率 (15%): 权益乘数, 资产周转率
        - Z-Score (15%): Altman Z-Score映射分

    参数:
        panel: data.build_credit_panel 返回的面板数据

    返回:
        dict，包含各维度得分和综合评分
    """
    from .data import compute_financial_ratios

    ratios = compute_financial_ratios(panel)

    # --- 盈利能力评分 ---
    roa = ratios.get("roa", 0) or 0
    roe = ratios.get("roe", 0) or 0
    net_margin = ratios.get("net_margin", 0) or 0
    # ROA>5%满分, 0-5%线性, <0零分
    profit_score = (
        _score_positive(roa, 0.05, 0.10) * 0.4 +
        _score_positive(roe, 0.10, 0.20) * 0.4 +
        _score_positive(net_margin, 0.10, 0.30) * 0.2
    ) * 100

    # --- 偿债能力评分 ---
    debt_ratio = ratios.get("debt_to_assets", 0.5) or 0.5
    int_cov = ratios.get("interest_coverage", 0) or 0
    # 资产负债率<30%满分, 30-60%线性, >80%零分
    debt_score = _score_inverted(debt_ratio, 0.30, 0.80) * 100
    # 利息保障倍数>5满分
    int_cov_score = _score_positive(int_cov, 3, 10) * 100
    solvency_score = debt_score * 0.6 + int_cov_score * 0.4

    # --- 流动性评分 ---
    current_ratio = ratios.get("current_ratio", 0) or 0
    # 流动比率>2满分, 1-2线性, <1零分
    liquidity_score = _score_positive(current_ratio, 1.0, 2.5) * 100

    # --- 运营效率评分 ---
    equity_mult = ratios.get("equity_multiplier", 0) or 0
    # 权益乘数1-3为佳, 越高杠杆越大风险越大
    if equity_mult <= 0:
        eff_score = 0
    elif equity_mult <= 2:
        eff_score = 100
    elif equity_mult <= 5:
        eff_score = (5 - equity_mult) / (5 - 2) * 100
    else:
        eff_score = 0

    # --- Z-Score映射分 ---
    lb = panel["latest_balance"]
    li = panel["latest_income"]
    z_result = compute_altman_zscore(
        current_assets=float(lb.get("current_assets", 0)),
        current_liab=float(lb.get("current_liab", 0)),
        total_assets=float(lb.get("total_assets", 0)),
        retained_earnings=float(lb.get("retained_earnings", 0)),
        ebit=float(li.get("total_profit", 0)) + abs(float(li.get("finance_expense", 0))),
        market_cap=panel["market_cap"],
        total_liab=panel["total_liab"],
        revenue=float(li.get("revenue", 0)),
        variant="public",
    )
    z = z_result["Z"]
    # Z-Score>5.85满分, 2.99-5.85线性, <1.81零分
    if z >= 5.85:
        z_score_val = 100
    elif z >= 2.99:
        z_score_val = (z - 2.99) / (5.85 - 2.99) * 100
    elif z >= 1.81:
        z_score_val = (z - 1.81) / (2.99 - 1.81) * 50  # 灰色区最高50
    else:
        z_score_val = 0

    # --- 综合评分（加权平均）---
    comprehensive = (
        profit_score * 0.25 +
        solvency_score * 0.25 +
        liquidity_score * 0.20 +
        eff_score * 0.15 +
        z_score_val * 0.15
    )

    return {
        "comprehensive_score": comprehensive,
        "profit_score": profit_score,
        "solvency_score": solvency_score,
        "liquidity_score": liquidity_score,
        "efficiency_score": eff_score,
        "zscore_mapped": z_score_val,
        "altman_z": z,
        "altman_zone": z_result["zone"],
        "altman_rating": zscore_to_rating(z),
        "financial_ratios": ratios,
    }


def _score_positive(value: float, threshold_low: float, threshold_high: float) -> float:
    """正值评分辅助函数：value>=threshold_high得1分，threshold_low~high线性，<low得0分。"""
    if value >= threshold_high:
        return 1.0
    elif value >= threshold_low:
        return (value - threshold_low) / (threshold_high - threshold_low)
    else:
        return 0.0


def _score_inverted(value: float, threshold_low: float, threshold_high: float) -> float:
    """反向评分辅助函数：value<=threshold_low得1分，low~high线性递减，>high得0分。"""
    if value <= threshold_low:
        return 1.0
    elif value <= threshold_high:
        return (threshold_high - value) / (threshold_high - threshold_low)
    else:
        return 0.0


# ============================================================
#  信用评分完整评估
# ============================================================

def scoring_assess(panel: dict) -> dict:
    """
    对单个公司执行完整的信用评分评估。

    参数:
        panel: data.build_credit_panel 返回的面板数据

    返回:
        dict，包含：
            - ticker              : 股票代码
            - name                : 公司简称
            - industry            : 行业
            - altman_z            : Z-Score值
            - altman_zone         : 区域判定
            - altman_rating       : Z-Score评级
            - comprehensive_score  : 综合评分（0-100）
            - profit_score        : 盈利能力评分
            - solvency_score      : 偿债能力评分
            - liquidity_score     : 流动性评分
            - efficiency_score    : 运营效率评分
            - financial_ratios    : 财务比率
            - rating              : 综合信用评级
    """
    from .data import compute_financial_ratios

    lb = panel["latest_balance"]
    li = panel["latest_income"]

    # EBIT = 利润总额 + 财务费用
    ebit = float(li.get("total_profit", 0)) + abs(float(li.get("finance_expense", 0)))

    # Altman Z-Score
    z_result = compute_altman_zscore(
        current_assets=float(lb.get("current_assets", 0)),
        current_liab=float(lb.get("current_liab", 0)),
        total_assets=float(lb.get("total_assets", 0)),
        retained_earnings=float(lb.get("retained_earnings", 0)),
        ebit=ebit,
        market_cap=panel["market_cap"],
        total_liab=panel["total_liab"],
        revenue=float(li.get("revenue", 0)),
        variant="public",
    )

    # 综合评分
    comp_result = compute_comprehensive_score(panel)

    # 综合评级
    comp_score = comp_result["comprehensive_score"]
    if comp_score >= 85:
        rating = "AAA"
    elif comp_score >= 75:
        rating = "AA"
    elif comp_score >= 65:
        rating = "A"
    elif comp_score >= 55:
        rating = "BBB"
    elif comp_score >= 45:
        rating = "BB"
    elif comp_score >= 35:
        rating = "B"
    else:
        rating = "CCC"

    return {
        "ticker": panel["ticker"],
        "name": panel["name"],
        "industry": panel["industry"],
        "altman_z": z_result["Z"],
        "altman_X1": z_result["X1"],
        "altman_X2": z_result["X2"],
        "altman_X3": z_result["X3"],
        "altman_X4": z_result["X4"],
        "altman_X5": z_result["X5"],
        "altman_zone": z_result["zone"],
        "altman_zone_detail": z_result["zone_detail"],
        "altman_rating": zscore_to_rating(z_result["Z"]),
        "comprehensive_score": comp_score,
        "profit_score": comp_result["profit_score"],
        "solvency_score": comp_result["solvency_score"],
        "liquidity_score": comp_result["liquidity_score"],
        "efficiency_score": comp_result["efficiency_score"],
        "rating": rating,
        "financial_ratios": comp_result["financial_ratios"],
    }


def scoring_assess_multi(panels: dict) -> pd.DataFrame:
    """
    对多个公司执行信用评分，返回汇总DataFrame。

    参数:
        panels: data.build_multi_company_panel 返回的面板字典

    返回:
        DataFrame，每行一个公司
    """
    results = []
    for ticker, panel in panels.items():
        try:
            res = scoring_assess(panel)
            # 展开财务比率为顶层列
            flat = {k: v for k, v in res.items() if k != "financial_ratios"}
            if "financial_ratios" in res:
                for rk, rv in res["financial_ratios"].items():
                    flat[f"ratio_{rk}"] = rv
            results.append(flat)
        except Exception as e:
            print(f"[警告] 信用评分 {ticker} 失败: {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("comprehensive_score", ascending=False).reset_index(drop=True)
    return df


# ============================================================
#  行业基准比较
# ============================================================

def industry_benchmark_comparison(scoring_results: pd.DataFrame) -> pd.DataFrame:
    """
    按行业汇总信用评分基准。

    参数:
        scoring_results: scoring_assess_multi 返回的DataFrame

    返回:
        DataFrame，按行业汇总的评分基准统计
    """
    if "industry" not in scoring_results.columns:
        return pd.DataFrame()

    stats = scoring_results.groupby("industry").agg(
        company_count=("ticker", "count"),
        z_mean=("altman_z", "mean"),
        z_median=("altman_z", "median"),
        score_mean=("comprehensive_score", "mean"),
        score_median=("comprehensive_score", "median"),
        score_min=("comprehensive_score", "min"),
        score_max=("comprehensive_score", "max"),
    ).reset_index()

    stats = stats.sort_values("score_mean", ascending=False).reset_index(drop=True)
    return stats
