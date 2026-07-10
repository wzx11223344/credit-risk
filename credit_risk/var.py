# -*- coding: utf-8 -*-
"""
信用VaR与组合损失模块
=======================

本模块实现信用风险度量体系，包括：
    - 信用VaR (Credit Value at Risk)
    - 预期损失 (Expected Loss, EL)
    - 意外损失 (Unexpected Loss, UL)
    - 经济资本 (Economic Capital, EC)
    - 边际风险贡献 (Marginal Risk Contribution)
    - 基于Copula的组合VaR
    - 蒙特卡洛模拟组合信用损失分布

核心公式:
    EL   = PD × LGD × EAD
    UL   = EAD × LGD × √(PD × (1-PD))         （单一资产）
    UL_p = √(ΣΣ ρ_ij × UL_i × UL_j)            （组合）
    EC   = UL_p × multiplier
    VaR  = Loss_α - EL                          （分位数法）
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ============================================================
#  单一资产信用风险度量
# ============================================================

def expected_loss(pd: float, lgd: float, ead: float) -> float:
    """
    计算预期损失 (Expected Loss, EL)。

    EL = PD × LGD × EAD

    参数:
        pd  : 违约概率 (Probability of Default)
        lgd : 违约损失率 (Loss Given Default) = 1 - 恢复率
        ead : 违约风险暴露 (Exposure at Default)

    返回:
        预期损失金额
    """
    return pd * lgd * ead


def unexpected_loss(pd: float, lgd: float, ead: float) -> float:
    """
    计算单一资产的意外损失 (Unexpected Loss, UL)。

    UL = EAD × LGD × √(PD × (1 - PD))

    这是损失分布的标准差（假设违约为伯努利事件，LGD确定）。

    参数:
        pd  : 违约概率
        lgd : 违约损失率
        ead : 违约风险暴露

    返回:
        意外损失金额
    """
    if pd <= 0 or pd >= 1:
        return 0.0
    return ead * lgd * np.sqrt(pd * (1 - pd))


def economic_capital_single(pd: float, lgd: float, ead: float,
                            multiplier: float = 3.0) -> float:
    """
    计算单一资产的经济资本。

    EC = UL × multiplier

    参数:
        pd         : 违约概率
        lgd        : 违约损失率
        ead        : 违约风险暴露
        multiplier : 资本乘数（通常2.5-3.0，对应99.9%置信度）

    返回:
        经济资本
    """
    ul = unexpected_loss(pd, lgd, ead)
    return ul * multiplier


# ============================================================
#  组合信用风险度量
# ============================================================

def portfolio_expected_loss(pds: np.ndarray, lgds: np.ndarray,
                             eads: np.ndarray) -> float:
    """
    计算组合预期损失。

    EL_portfolio = Σ_i PD_i × LGD_i × EAD_i

    参数:
        pds  : 违约概率向量 (n,)
        lgds : 违约损失率向量 (n,)
        eads : 违约风险暴露向量 (n,)

    返回:
        组合预期损失
    """
    pds = np.asarray(pds, dtype=float)
    lgds = np.asarray(lgds, dtype=float)
    eads = np.asarray(eads, dtype=float)
    return float(np.sum(pds * lgds * eads))


def portfolio_unexpected_loss(pds: np.ndarray, lgds: np.ndarray,
                               eads: np.ndarray,
                               corr_matrix: np.ndarray = None) -> float:
    """
    计算组合意外损失。

    UL_portfolio = √(Σ_i Σ_j ρ_ij × UL_i × UL_j)

    其中 UL_i = EAD_i × LGD_i × √(PD_i × (1-PD_i)) 为单一资产意外损失，
    ρ_ij 为违约相关性矩阵。

    参数:
        pds         : 违约概率向量 (n,)
        lgds        : 违约损失率向量 (n,)
        eads        : 违约风险暴露向量 (n,)
        corr_matrix : 违约相关矩阵 (n x n)，None则假设不相关

    返回:
        组合意外损失
    """
    pds = np.asarray(pds, dtype=float)
    lgds = np.asarray(lgds, dtype=float)
    eads = np.asarray(eads, dtype=float)
    n = len(pds)

    # 单一资产UL
    uls = np.array([
        unexpected_loss(pds[i], lgds[i], eads[i]) for i in range(n)
    ])

    if corr_matrix is None:
        # 不相关：UL_p = sqrt(sum(UL_i^2))
        return float(np.sqrt(np.sum(uls ** 2)))

    corr_matrix = np.asarray(corr_matrix, dtype=float)
    # UL_p = sqrt(UL' × R × UL)
    portfolio_ul = np.sqrt(uls @ corr_matrix @ uls)
    return float(portfolio_ul)


def marginal_risk_contribution(pds: np.ndarray, lgds: np.ndarray,
                                eads: np.ndarray,
                                corr_matrix: np.ndarray,
                                i: int) -> float:
    """
    计算第i个资产的边际风险贡献。

    MRC_i = (Σ_j ρ_ij × UL_i × UL_j) / UL_portfolio

    边际风险贡献表示增加1单位该资产暴露对组合UL的增量影响。

    参数:
        pds         : 违约概率向量
        lgds        : 违约损失率向量
        eads        : 违约风险暴露向量
        corr_matrix : 违约相关矩阵
        i           : 资产索引

    返回:
        边际风险贡献
    """
    pds = np.asarray(pds, dtype=float)
    lgds = np.asarray(lgds, dtype=float)
    eads = np.asarray(eads, dtype=float)
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    n = len(pds)

    uls = np.array([
        unexpected_loss(pds[j], lgds[j], eads[j]) for j in range(n)
    ])

    portfolio_ul = np.sqrt(uls @ corr_matrix @ uls)
    if portfolio_ul <= 0:
        return 0.0

    # MRC_i = sum_j(rho_ij * UL_i * UL_j) / UL_portfolio
    mrc = np.sum(corr_matrix[i, :] * uls[i] * uls) / portfolio_ul
    return float(mrc)


def all_risk_contributions(pds: np.ndarray, lgds: np.ndarray,
                            eads: np.ndarray,
                            corr_matrix: np.ndarray) -> pd.DataFrame:
    """
    计算所有资产的边际风险贡献。

    参数:
        pds         : 违约概率向量
        lgds        : 违约损失率向量
        eads        : 违约风险暴露向量
        corr_matrix : 违约相关矩阵

    返回:
        DataFrame，包含每个资产的EL、UL、MRC
    """
    pds = np.asarray(pds, dtype=float)
    lgds = np.asarray(lgds, dtype=float)
    eads = np.asarray(eads, dtype=float)
    n = len(pds)

    records = []
    for i in range(n):
        el_i = expected_loss(pds[i], lgds[i], eads[i])
        ul_i = unexpected_loss(pds[i], lgds[i], eads[i])
        mrc_i = marginal_risk_contribution(pds, lgds, eads, corr_matrix, i)
        records.append({
            "asset_index": i,
            "PD": pds[i],
            "LGD": lgds[i],
            "EAD": eads[i],
            "EL": el_i,
            "UL": ul_i,
            "MRC": mrc_i,
            "EL_pct": el_i / eads[i] if eads[i] > 0 else 0,
        })

    return pd.DataFrame(records)


# ============================================================
#  违约相关性
# ============================================================

def default_correlation_matrix(pd_vector: np.ndarray,
                                asset_corr: np.ndarray) -> np.ndarray:
    """
    从资产相关性矩阵推导违约相关性矩阵。

    在Gaussian Copula框架下：
    ρ_D_ij = [Φ_2(Φ^{-1}(PD_i), Φ^{-1}(PD_j); ρ_A) - PD_i·PD_j]
             / [√(PD_i(1-PD_i)) · √(PD_j(1-PD_j))]

    其中 Φ_2 为二元标准正态联合CDF，ρ_A 为资产相关系数。

    参数:
        pd_vector  : 违约概率向量 (n,)
        asset_corr : 资产相关矩阵 (n x n)

    返回:
        违约相关矩阵 (n x n)
    """
    from scipy.stats import multivariate_normal as mvn

    pds = np.asarray(pd_vector, dtype=float)
    n = len(pds)
    asset_corr = np.asarray(asset_corr, dtype=float)

    # 逆正态分位数
    q = norm.ppf(pds)

    # 违约相关性矩阵
    rho_d = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            # 二元正态联合CDF
            cov_ij = np.array([
                [1.0, asset_corr[i, j]],
                [asset_corr[i, j], 1.0]
            ])
            try:
                joint = mvn.cdf([q[i], q[j]], mean=[0, 0], cov=cov_ij)
            except Exception:
                joint = pds[i] * pds[j]  # 退化情况

            denom = np.sqrt(pds[i] * (1 - pds[i]) * pds[j] * (1 - pds[j]))
            if denom > 0:
                rho_d[i, j] = (joint - pds[i] * pds[j]) / denom
                rho_d[j, i] = rho_d[i, j]
            else:
                rho_d[i, j] = 0.0
                rho_d[j, i] = 0.0

    return rho_d


# ============================================================
#  信用VaR（解析法）
# ============================================================

def credit_var_analytical(pds: np.ndarray, lgds: np.ndarray,
                           eads: np.ndarray,
                           corr_matrix: np.ndarray,
                           confidence: float = 0.999) -> dict:
    """
    解析法计算信用VaR（基于正态近似）。

    假设组合损失服从正态分布：
    Loss ~ N(EL, UL²)
    VaR_α = EL + Z_α × UL
    Credit VaR = VaR_α - EL = Z_α × UL

    参数:
        pds         : 违约概率向量
        lgds        : 违约损失率向量
        eads        : 违约风险暴露向量
        corr_matrix : 违约相关矩阵
        confidence  : 置信水平，默认0.999（99.9%）

    返回:
        dict，包含：
            - EL          : 预期损失
            - UL          : 意外损失
            - total_var   : 总VaR（含EL）
            - credit_var  : 信用VaR（=总VaR - EL）
            - ec          : 经济资本（=信用VaR）
            - confidence  : 置信水平
            - z_score     : 分位数
    """
    el = portfolio_expected_loss(pds, lgds, eads)
    ul = portfolio_unexpected_loss(pds, lgds, eads, corr_matrix)

    z_score = norm.ppf(confidence)
    total_var = el + z_score * ul
    credit_var = total_var - el

    return {
        "EL": el,
        "UL": ul,
        "total_var": total_var,
        "credit_var": credit_var,
        "ec": credit_var,  # 经济资本 = 信用VaR
        "confidence": confidence,
        "z_score": z_score,
    }


# ============================================================
#  蒙特卡洛模拟组合信用损失分布
# ============================================================

def monte_carlo_credit_loss(
    pds: np.ndarray,
    lgds: np.ndarray,
    eads: np.ndarray,
    corr_matrix: np.ndarray,
    n_simulations: int = 100000,
    confidence: float = 0.999,
    random_state: int = 42,
) -> dict:
    """
    蒙特卡洛模拟组合信用损失分布。

    使用Gaussian Copula模型生成相关的违约事件：
        1. 从多元正态分布生成相关因子
        2. 转为均匀分布
        3. 若 u_i < PD_i 则发生违约
        4. 损失 = Σ EAD_i × LGD_i × I(违约_i)

    参数:
        pds         : 违约概率向量 (n,)
        lgds        : 违约损失率向量 (n,)
        eads        : 违约风险暴露向量 (n,)
        corr_matrix : 资产相关矩阵 (n x n)（不是违约相关矩阵）
        n_simulations: 模拟次数，默认100000
        confidence  : VaR置信水平，默认0.999
        random_state: 随机种子（固定为42保证可复现）

    返回:
        dict，包含：
            - losses         : 损失分布数组 (n_simulations,)
            - EL_sim         : 模拟预期损失
            - UL_sim         : 模拟意外损失
            - var_total      : 总VaR（含EL）
            - credit_var     : 信用VaR（=VaR - EL）
            - ec             : 经济资本
            - worst_loss     : 最坏损失
            - best_loss      : 最好损失（通常为0）
            - default_rate   : 平均违约率
            - confidence     : 置信水平
            - n_simulations  : 模拟次数
    """
    pds = np.asarray(pds, dtype=float)
    lgds = np.asarray(lgds, dtype=float)
    eads = np.asarray(eads, dtype=float)
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    n_assets = len(pds)

    # 确保相关矩阵正定
    corr_matrix = _make_positive_definite(corr_matrix)

    # 生成相关正态随机数
    rng = np.random.RandomState(random_state)
    z = rng.multivariate_normal(
        mean=np.zeros(n_assets), cov=corr_matrix, size=n_simulations
    )

    # 转为均匀分布
    u = norm.cdf(z)

    # 违约指示：u < PD 则违约
    default_indicators = (u < pds[np.newaxis, :]).astype(float)

    # 每次模拟的损失
    losses = default_indicators @ (lgds * eads)

    # 统计量
    el_sim = float(np.mean(losses))
    ul_sim = float(np.std(losses))

    # VaR
    var_total = float(np.percentile(losses, confidence * 100))
    credit_var = var_total - el_sim
    ec = credit_var

    # 违约率统计
    default_count = default_indicators.sum(axis=1)
    default_rate = float(np.mean(default_count > 0))

    return {
        "losses": losses,
        "EL_sim": el_sim,
        "UL_sim": ul_sim,
        "var_total": var_total,
        "credit_var": credit_var,
        "ec": ec,
        "worst_loss": float(np.max(losses)),
        "best_loss": float(np.min(losses)),
        "default_rate": default_rate,
        "confidence": confidence,
        "n_simulations": n_simulations,
    }


def _make_positive_definite(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """确保矩阵正定。"""
    m = mat.copy()
    np.fill_diagonal(m, 1.0)
    eigvals, eigvecs = np.linalg.eigh(m)
    eigvals = np.maximum(eigvals, eps)
    m = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(m))
    m = m / np.outer(d, d)
    np.fill_diagonal(m, 1.0)
    return m


# ============================================================
#  基于Copula的组合信用VaR
# ============================================================

def copula_portfolio_var(
    copula,
    pds: np.ndarray,
    lgds: np.ndarray,
    eads: np.ndarray,
    n_simulations: int = 100000,
    confidence: float = 0.999,
    random_state: int = 42,
) -> dict:
    """
    基于Copula的组合信用VaR蒙特卡洛模拟。

    使用任意Copula实例（Gaussian/Student-t/Clayton）生成
    相关的违约事件，构建损失分布并计算VaR。

    参数:
        copula        : Copula实例（GaussianCopula/StudentTCopula）
        pds           : 违约概率向量
        lgds          : 违约损失率向量
        eads          : 违约风险暴露向量
        n_simulations : 模拟次数
        confidence    : 置信水平
        random_state  : 随机种子

    返回:
        dict，包含损失分布和VaR统计量
    """
    pds = np.asarray(pds, dtype=float)
    lgds = np.asarray(lgds, dtype=float)
    eads = np.asarray(eads, dtype=float)

    # 从Copula生成均匀分布样本
    u = copula.simulate(n_simulations, random_state=random_state)

    # 违约指示
    default_indicators = (u < pds[np.newaxis, :]).astype(float)

    # 损失
    losses = default_indicators @ (lgds * eads)

    # 统计
    el = float(np.mean(losses))
    ul = float(np.std(losses))
    var_total = float(np.percentile(losses, confidence * 100))
    credit_var = var_total - el

    return {
        "losses": losses,
        "EL": el,
        "UL": ul,
        "var_total": var_total,
        "credit_var": credit_var,
        "ec": credit_var,
        "confidence": confidence,
        "n_simulations": n_simulations,
        "copula_type": type(copula).__name__,
    }


# ============================================================
#  组合信用风险完整评估
# ============================================================

def portfolio_credit_assessment(
    kmv_results: pd.DataFrame,
    copula_result: dict,
    eads: np.ndarray = None,
    lgds: np.ndarray = None,
    confidence: float = 0.999,
    n_simulations: int = 100000,
    random_state: int = 42,
) -> dict:
    """
    组合信用风险完整评估。

    整合KMV模型、Copula依赖结构和蒙特卡洛模拟，
    输出完整的组合信用风险度量。

    参数:
        kmv_results   : KMV模型结果DataFrame
        copula_result : Copula分析结果（来自copula.portfolio_default_correlation）
        eads          : 违约风险暴露向量，None则按等权重分配
        lgds          : 违约损失率向量，None则默认0.55（即恢复率45%）
        confidence    : 置信水平，默认0.999
        n_simulations : 蒙特卡洛模拟次数
        random_state  : 随机种子

    返回:
        dict，包含完整评估结果
    """
    pds = copula_result["marginal_pds"]
    n = len(pds)
    tickers = copula_result["tickers"]

    # 默认参数
    if lgds is None:
        lgds = np.full(n, 0.55)  # 默认LGD=55%（恢复率45%）
    if eads is None:
        # 默认等权重暴露1亿元
        eads = np.full(n, 1e8)

    # 资产相关矩阵（从Copula相关矩阵获取）
    asset_corr = copula_result["corr_matrix"]

    # 违约相关矩阵
    rho_d = default_correlation_matrix(pds, asset_corr)

    # 解析法VaR
    var_analytic = credit_var_analytical(
        pds, lgds, eads, rho_d, confidence=confidence
    )

    # 蒙特卡洛模拟
    mc_result = monte_carlo_credit_loss(
        pds, lgds, eads, asset_corr,
        n_simulations=n_simulations,
        confidence=confidence,
        random_state=random_state,
    )

    # 风险贡献
    contributions = all_risk_contributions(pds, lgds, eads, rho_d)
    contributions.insert(0, "ticker", tickers)
    contributions.insert(1, "name", kmv_results["name"].values[:n])

    # 组合汇总
    portfolio_summary = {
        "n_assets": n,
        "total_ead": float(np.sum(eads)),
        "total_el": var_analytic["EL"],
        "total_ul": var_analytic["UL"],
        "credit_var": var_analytic["credit_var"],
        "ec": var_analytic["ec"],
        "mc_el": mc_result["EL_sim"],
        "mc_ul": mc_result["UL_sim"],
        "mc_credit_var": mc_result["credit_var"],
        "mc_worst_loss": mc_result["worst_loss"],
        "confidence": confidence,
        "tickers": tickers,
    }

    return {
        "portfolio_summary": portfolio_summary,
        "var_analytic": var_analytic,
        "mc_result": mc_result,
        "contributions": contributions,
        "default_corr_matrix": rho_d,
        "asset_corr_matrix": asset_corr,
        "pds": pds,
        "lgds": lgds,
        "eads": eads,
    }
