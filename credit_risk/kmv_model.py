# -*- coding: utf-8 -*-
"""
KMV-Merton 结构化信用风险模型
================================

本模块实现经典的KMV-Merton结构化违约模型，核心思想：

    将公司股权视为以公司资产为标的、以违约点为行权价的看涨期权。

核心方程：
    (1) 股权价值: E = V·N(d1) - D·e^(-rT)·N(d2)    [Black-Scholes看涨期权]
    (2) 股权波动: σ_E·E = N(d1)·σ_A·V              [Ito引理]

其中:
    d1 = [ln(V/D) + (r + 0.5·σ_A²)·T] / (σ_A·√T)
    d2 = d1 - σ_A·√T

通过迭代求解资产价值 V 和资产波动率 σ_A，进而计算:
    - 违约距离 (Distance to Default, DD)
    - 违约概率 (Expected Default Frequency, EDF)
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# 尝试获取国债收益率作为无风险利率
try:
    import akshare as ak
except ImportError:
    ak = None


def get_risk_free_rate() -> float:
    """
    获取中国1年期国债到期收益率作为无风险利率。

    使用akshare的国债收益率接口。若获取失败，返回默认值2.5%。

    返回:
        无风险利率（小数形式），如 0.025 表示 2.5%
    """
    if ak is None:
        return 0.025
    try:
        # 获取中国国债收益率曲线
        df = ak.bond_china_yield(start_date="20240101", end_date="20240601")
        if df is not None and len(df) > 0:
            # 寻找1年期国债收益率列
            for col in df.columns:
                if "1年" in str(col) or "1 年" in str(col):
                    rate = pd.to_numeric(df[col].dropna().iloc[-1], errors="coerce")
                    if not np.isnan(rate):
                        return float(rate) / 100.0  # 转为小数
    except Exception:
        pass
    return 0.025  # 默认2.5%


# 需要pandas
import pandas as pd


# ============================================================
#  Black-Scholes-Merton 期权定价相关函数
# ============================================================

def _d1_d2(V: float, D: float, r: float, sigma_A: float, T: float):
    """
    计算BSM模型中的d1和d2。

    参数:
        V      : 资产价值
        D      : 违约点（负债面值）
        r      : 无风险利率
        sigma_A: 资产波动率
        T      : 时间期限（年）

    返回:
        (d1, d2) 元组
    """
    if V <= 0 or D <= 0 or sigma_A <= 0 or T <= 0:
        return np.nan, np.nan

    ln_ratio = np.log(V / D)
    d1 = (ln_ratio + (r + 0.5 * sigma_A ** 2) * T) / (sigma_A * np.sqrt(T))
    d2 = d1 - sigma_A * np.sqrt(T)
    return d1, d2


def bsm_equity_value(V: float, D: float, r: float, sigma_A: float, T: float) -> float:
    """
    Black-Scholes-Merton模型下的股权价值。

    E = V·N(d1) - D·e^(-rT)·N(d2)

    参数:
        V      : 资产价值
        D      : 违约点
        r      : 无风险利率
        sigma_A: 资产波动率
        T      : 时间期限（年）

    返回:
        股权价值 E
    """
    d1, d2 = _d1_d2(V, D, r, sigma_A, T)
    if np.isnan(d1):
        return np.nan
    E = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)
    return max(E, 1e-8)  # 股权价值下限


def bsm_equity_volatility(V: float, E: float, D: float, r: float,
                          sigma_A: float, T: float) -> float:
    """
    由BSM模型隐含的股权波动率。

    σ_E = (V · N(d1) · σ_A) / E

    参数:
        V      : 资产价值
        E      : 股权价值（市场值）
        D      : 违约点
        r      : 无风险利率
        sigma_A: 资产波动率
        T      : 时间期限（年）

    返回:
        隐含股权波动率 σ_E
    """
    d1, d2 = _d1_d2(V, D, r, sigma_A, T)
    if np.isnan(d1) or E <= 0:
        return np.nan
    nd1 = norm.cdf(d1)
    if nd1 <= 0:
        return np.nan
    sigma_E = (V * nd1 * sigma_A) / E
    return sigma_E


# ============================================================
#  迭代求解资产价值和波动率
# ============================================================

def solve_asset_value_and_volatility(
    equity_value: float,
    equity_volatility: float,
    default_point: float,
    risk_free_rate: float = 0.025,
    T: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict:
    """
    迭代求解公司资产价值和资产波动率。

    算法（固定点迭代 + Brent一维求根）：
        1. 初始化 σ_A = σ_E * E / (E + D)
        2. 给定 σ_A，用Brent法从BSM方程求解 V
        3. 由隐含股权波动率公式更新 σ_A
        4. 重复直到 σ_A 收敛

    参数:
        equity_value      : 股权市场价值（=总市值）
        equity_volatility : 股价年化波动率
        default_point     : 违约点 = 短期负债 + 0.5 * 长期负债
        risk_free_rate    : 无风险利率，默认0.025
        T                 : 时间期限（年），默认1.0
        max_iter          : 最大迭代次数，默认500
        tol               : 收敛容差，默认1e-8

    返回:
        dict，包含：
            - asset_value     : 资产价值 V
            - asset_volatility: 资产波动率 σ_A
            - equity_value    : 股权价值 E（输入）
            - equity_vol      : 股权波动率 σ_E（输入）
            - default_point   : 违约点 D
            - r               : 无风险利率
            - T               : 时间期限
            - converged       : 是否收敛
            - iterations      : 迭代次数
    """
    E = equity_value
    sigma_E = equity_volatility
    D = default_point
    r = risk_free_rate

    # 边界检查
    if E <= 0:
        raise ValueError("股权价值必须为正数")
    if sigma_E <= 0:
        raise ValueError("股权波动率必须为正数")
    if D <= 0:
        # 无负债时，资产价值=股权价值，资产波动率=股权波动率
        return {
            "asset_value": E,
            "asset_volatility": sigma_E,
            "equity_value": E,
            "equity_vol": sigma_E,
            "default_point": D,
            "r": r,
            "T": T,
            "converged": True,
            "iterations": 0,
        }

    # 初始猜测：σ_A = σ_E * E / (E + D)
    sigma_A = sigma_E * E / (E + D)
    if sigma_A <= 0:
        sigma_A = sigma_E * 0.5  # 兜底

    converged = False
    prev_sigma_A = sigma_A

    for iteration in range(max_iter):
        # --- 第1步：给定 σ_A，从BSM方程求解 V ---
        # 方程: f(V) = BSM_E(V, σ_A) - E_market = 0
        # V 的范围: (E, E + D*exp(-rT)) 大致，但更宽泛地搜索
        V_min = max(E * 0.01, 1e-4)
        V_max = E + D * np.exp(-r * T) * 10  # 上界放宽

        def _eq(V):
            return bsm_equity_value(V, D, r, sigma_A, T) - E

        # 检查符号是否改变（Brent法要求）
        f_min = _eq(V_min)
        f_max = _eq(V_max)

        if f_min * f_max > 0:
            # 如果同号，调整边界
            if f_min > 0:
                V = V_min
            else:
                V = V_max
        else:
            # 使用Brent法求根
            try:
                V = brentq(_eq, V_min, V_max, xtol=tol, maxiter=200)
            except ValueError:
                # 求根失败，用V_max
                V = V_max if abs(f_max) < abs(f_min) else V_min

        # --- 第2步：由隐含股权波动率更新 σ_A ---
        sigma_E_implied = bsm_equity_volatility(V, E, D, r, sigma_A, T)

        if np.isnan(sigma_E_implied) or sigma_E_implied <= 0:
            break

        # 更新规则：σ_A_new = σ_A * σ_E_target / σ_E_implied
        # 这保证 σ_E_implied -> σ_E_target 时 σ_A 收敛
        sigma_A_new = sigma_A * (sigma_E / sigma_E_implied)

        # 安全检查：防止异常值
        if sigma_A_new <= 0 or sigma_A_new > 10:
            sigma_A_new = sigma_A  # 不更新

        # --- 第3步：检查收敛 ---
        if abs(sigma_A_new - prev_sigma_A) < tol:
            sigma_A = sigma_A_new
            converged = True
            break

        sigma_A = sigma_A_new
        prev_sigma_A = sigma_A

    # 最终重新求解V（用最终的sigma_A）
    V_min = max(E * 0.01, 1e-4)
    V_max = E + D * np.exp(-r * T) * 10

    def _eq_final(V):
        return bsm_equity_value(V, D, r, sigma_A, T) - E

    f_min = _eq_final(V_min)
    f_max = _eq_final(V_max)

    if f_min * f_max <= 0:
        try:
            V = brentq(_eq_final, V_min, V_max, xtol=tol, maxiter=200)
        except ValueError:
            pass

    return {
        "asset_value": V,
        "asset_volatility": sigma_A,
        "equity_value": E,
        "equity_vol": sigma_E,
        "default_point": D,
        "r": r,
        "T": T,
        "converged": converged,
        "iterations": iteration + 1,
    }


# ============================================================
#  违约距离与违约概率
# ============================================================

def distance_to_default(
    asset_value: float,
    default_point: float,
    asset_volatility: float,
    drift: float = 0.0,
    T: float = 1.0,
) -> float:
    """
    计算违约距离 (Distance to Default, DD)。

    DD = [ln(V/D) + (μ - 0.5·σ_A²)·T] / (σ_A·√T)

    其中：
        V   = 资产价值
        D   = 违约点
        μ   = 资产漂移率（预期收益率）
        σ_A = 资产波动率
        T   = 时间期限

    DD的经济含义：公司资产价值距离违约点有多少个标准差。
    DD越大，违约可能性越小。

    参数:
        asset_value     : 资产价值 V
        default_point   : 违约点 D
        asset_volatility: 资产波动率 σ_A
        drift           : 资产漂移率 μ，默认0（风险中性下可设为无风险利率）
        T               : 时间期限（年），默认1.0

    返回:
        违约距离 DD
    """
    if asset_value <= 0 or default_point <= 0 or asset_volatility <= 0 or T <= 0:
        return np.nan

    ln_ratio = np.log(asset_value / default_point)
    dd = (ln_ratio + (drift - 0.5 * asset_volatility ** 2) * T) / \
         (asset_volatility * np.sqrt(T))
    return dd


def expected_default_frequency(dd: float) -> float:
    """
    计算预期违约频率 (Expected Default Frequency, EDF)。

    EDF = N(-DD) = Φ(-DD)

    即在风险中性框架下，资产价值在T时刻低于违约点的概率。

    参数:
        dd: 违约距离

    返回:
        违约概率 EDF（0~1之间）
    """
    if np.isnan(dd):
        return np.nan
    return norm.cdf(-dd)


# ============================================================
#  KMV模型完整评估
# ============================================================

def kmv_assess(
    panel: dict,
    risk_free_rate: float = None,
    T: float = 1.0,
    drift: float = None,
) -> dict:
    """
    对单个公司执行完整的KMV模型评估。

    参数:
        panel           : data.build_credit_panel 返回的面板数据
        risk_free_rate  : 无风险利率，None则自动获取
        T               : 时间期限（年），默认1.0
        drift           : 资产漂移率，None则使用无风险利率（风险中性）

    返回:
        dict，包含KMV模型全部输出：
            - ticker          : 股票代码
            - name            : 公司简称
            - industry        : 行业
            - equity_value    : 股权价值
            - equity_vol      : 股权波动率
            - default_point   : 违约点
            - asset_value     : 资产价值
            - asset_volatility: 资产波动率
            - dd              : 违约距离
            - edf             : 违约概率
            - leverage_ratio  : 杠杆率 (D/V)
            - converged       : 迭代是否收敛
            - iterations      : 迭代次数
            - r               : 无风险利率
            - T               : 时间期限
    """
    # 获取无风险利率
    if risk_free_rate is None:
        risk_free_rate = get_risk_free_rate()

    # 漂移率默认使用无风险利率（风险中性测度）
    if drift is None:
        drift = risk_free_rate

    # 从面板提取输入
    E = panel["equity_value"]
    sigma_E = panel["stock_volatility"]
    D = panel["default_point"]

    # 迭代求解
    result = solve_asset_value_and_volatility(
        equity_value=E,
        equity_volatility=sigma_E,
        default_point=D,
        risk_free_rate=risk_free_rate,
        T=T,
    )

    # 计算违约距离
    dd = distance_to_default(
        asset_value=result["asset_value"],
        default_point=D,
        asset_volatility=result["asset_volatility"],
        drift=drift,
        T=T,
    )

    # 计算违约概率
    edf = expected_default_frequency(dd)

    # 杠杆率
    leverage = D / result["asset_value"] if result["asset_value"] > 0 else np.nan

    return {
        "ticker": panel["ticker"],
        "name": panel["name"],
        "industry": panel["industry"],
        "equity_value": E,
        "equity_vol": sigma_E,
        "default_point": D,
        "asset_value": result["asset_value"],
        "asset_volatility": result["asset_volatility"],
        "dd": dd,
        "edf": edf,
        "leverage_ratio": leverage,
        "converged": result["converged"],
        "iterations": result["iterations"],
        "r": risk_free_rate,
        "T": T,
    }


def kmv_assess_multi(panels: dict, risk_free_rate: float = None,
                     T: float = 1.0) -> pd.DataFrame:
    """
    对多个公司执行KMV模型评估，返回汇总DataFrame。

    参数:
        panels         : data.build_multi_company_panel 返回的面板字典
        risk_free_rate : 无风险利率，None则自动获取
        T              : 时间期限（年），默认1.0

    返回:
        DataFrame，每行一个公司，包含KMV模型主要输出
    """
    if risk_free_rate is None:
        risk_free_rate = get_risk_free_rate()

    results = []
    for ticker, panel in panels.items():
        try:
            res = kmv_assess(panel, risk_free_rate=risk_free_rate, T=T)
            results.append(res)
        except Exception as e:
            print(f"[警告] KMV评估 {ticker} 失败: {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # 按违约距离排序（DD越小风险越大）
    df = df.sort_values("dd", ascending=False).reset_index(drop=True)
    return df


def dd_industry_comparison(kmv_results: pd.DataFrame) -> pd.DataFrame:
    """
    违约距离行业比较。

    按行业分组统计违约距离均值、中位数、最小值。

    参数:
        kmv_results: kmv_assess_multi 返回的DataFrame

    返回:
        DataFrame，按行业汇总的违约距离统计
    """
    if "industry" not in kmv_results.columns or "dd" not in kmv_results.columns:
        return pd.DataFrame()

    stats = kmv_results.groupby("industry").agg(
        company_count=("ticker", "count"),
        dd_mean=("dd", "mean"),
        dd_median=("dd", "median"),
        dd_min=("dd", "min"),
        dd_max=("dd", "max"),
        edf_mean=("edf", "mean"),
    ).reset_index()

    stats = stats.sort_values("dd_mean", ascending=False).reset_index(drop=True)
    return stats
