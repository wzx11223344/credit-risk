# -*- coding: utf-8 -*-
"""
Copula 依赖结构建模模块
=========================

本模块实现多种Copula函数用于信用风险中的违约相关性建模：

    - Gaussian Copula   （高斯Copula，全局相依）
    - Student-t Copula  （t-Copula，捕捉厚尾相依）
    - Clayton Copula    （Clayton Copula，下尾相依）

核心功能：
    - Kendall秩相关系数估计
    - Copula参数拟合（最大似然估计 / 矩估计）
    - 联合违约概率计算
    - 资产组合信用风险（考虑违约相关性）
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm, t as t_dist, multivariate_normal


# ============================================================
#  Kendall秩相关系数
# ============================================================

def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算两个序列的Kendall秩相关系数 τ。

    Kendall's tau衡量两个随机变量单调关系的强度和方向。
    取值范围 [-1, 1]。

    参数:
        x: 序列1（一维数组）
        y: 序列2（一维数组）

    返回:
        Kendall's tau值
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    if len(x) < 2:
        return 0.0

    tau, _ = stats.kendalltau(x, y)
    return float(tau) if not np.isnan(tau) else 0.0


def kendall_tau_matrix(data: np.ndarray) -> np.ndarray:
    """
    计算多变量数据的Kendall秩相关系数矩阵。

    参数:
        data: (n_samples, n_features) 二维数组

    返回:
        (n_features, n_features) Kendall相关矩阵
    """
    df = pd.DataFrame(data)
    corr = df.corr(method="kendall").values
    return corr


# ============================================================
#  Copula参数与Kendall's tau的关系
# ============================================================

def gaussian_corr_from_tau(tau: float) -> float:
    """
    从Kendall's tau推导Gaussian Copula的相关系数。

    关系: ρ = sin(π·τ / 2)

    参数:
        tau: Kendall's tau

    返回:
        线性相关系数 ρ
    """
    return np.sin(np.pi * tau / 2.0)


def clayton_theta_from_tau(tau: float) -> float:
    """
    从Kendall's tau推导Clayton Copula的参数θ。

    关系: τ = θ / (θ + 2)  =>  θ = 2τ / (1 - τ)

    约束: θ > 0 （正相依），τ必须在 [0, 1) 范围内。

    参数:
        tau: Kendall's tau

    返回:
        Clayton Copula参数 θ
    """
    if tau >= 1.0:
        return 50.0  # 极强相依时的上限
    if tau <= 0:
        return 1e-6  # 最小值（趋近独立）
    return 2.0 * tau / (1.0 - tau)


# ============================================================
#  Gaussian Copula
# ============================================================

class GaussianCopula:
    """
    Gaussian Copula（高斯Copula）。

    C(u1,...,un) = Φ_Σ(Φ^{-1}(u1), ..., Φ^{-1}(un))

    参数:
        corr_matrix: 相关系数矩阵 (n x n)
    """

    def __init__(self, corr_matrix: np.ndarray):
        self.corr_matrix = np.asarray(corr_matrix, dtype=float)
        self.dim = self.corr_matrix.shape[0]
        # 确保相关矩阵正定
        self.corr_matrix = self._make_positive_definite(self.corr_matrix)
        self.mvn = multivariate_normal(
            mean=np.zeros(self.dim), cov=self.corr_matrix, allow_singular=False
        )

    @staticmethod
    def _make_positive_definite(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """确保矩阵正定（通过添加对角微扰）。"""
        m = mat.copy()
        # 对角线置1
        np.fill_diagonal(m, 1.0)
        # 特征值分解，将负特征值提升
        eigvals, eigvecs = np.linalg.eigh(m)
        eigvals = np.maximum(eigvals, eps)
        m = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # 重新归一化对角线为1
        d = np.sqrt(np.diag(m))
        m = m / np.outer(d, d)
        np.fill_diagonal(m, 1.0)
        return m

    def cdf(self, u: np.ndarray) -> float:
        """
        计算Gaussian Copula的CDF值。

        C(u1,...,un) = Φ_Σ(Φ^{-1}(u1), ..., Φ^{-1}(un))

        参数:
            u: 均匀分布分位数向量 (n,)，每个元素在(0,1)

        返回:
            Copula CDF值
        """
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 1e-10, 1 - 1e-10)
        z = norm.ppf(u)  # 逆标准正态
        return float(self.mvn.cdf(z))

    def simulate(self, n_samples: int, random_state: int = 42) -> np.ndarray:
        """
        从Gaussian Copula生成样本。

        参数:
            n_samples  : 样本数
            random_state: 随机种子（保证可复现）

        返回:
            (n_samples, dim) 的均匀分布样本
        """
        rng = np.random.RandomState(random_state)
        z = rng.multivariate_normal(
            mean=np.zeros(self.dim), cov=self.corr_matrix, size=n_samples
        )
        u = norm.cdf(z)
        return u

    @classmethod
    def fit_from_data(cls, data: np.ndarray) -> "GaussianCopula":
        """
        从数据拟合Gaussian Copula。

        步骤:
            1. 将数据转为均匀分布（经验CDF或参数CDF）
            2. 计算Kendall's tau矩阵
            3. 由 τ -> ρ = sin(πτ/2) 构建相关矩阵

        参数:
            data: (n_samples, n_features) 数据矩阵

        返回:
            拟合后的GaussianCopula实例
        """
        data = np.asarray(data, dtype=float)
        n, d = data.shape

        # 计算Kendall tau矩阵
        tau_mat = kendall_tau_matrix(data)

        # 转换为相关矩阵
        corr = np.sin(np.pi * tau_mat / 2.0)

        return cls(corr)


# ============================================================
#  Student-t Copula
# ============================================================

class StudentTCopula:
    """
    Student-t Copula（t-Copula）。

    C(u1,...,un) = T_{Σ,ν}(t_ν^{-1}(u1), ..., t_ν^{-1}(un))

    比Gaussian Copula更能捕捉尾部相依性。

    参数:
        corr_matrix: 相关系数矩阵 (n x n)
        df         : 自由度（越小尾部越厚）
    """

    def __init__(self, corr_matrix: np.ndarray, df: float = 5.0):
        self.corr_matrix = np.asarray(corr_matrix, dtype=float)
        self.dim = self.corr_matrix.shape[0]
        self.corr_matrix = GaussianCopula._make_positive_definite(self.corr_matrix)
        self.df = df

    def _mv_t_cdf(self, x: np.ndarray) -> float:
        """
        多元t分布CDF的蒙特卡洛近似计算。

        多元t: X = Z / sqrt(W/df), Z~N(0,Σ), W~chi2(df)
        CDF(x) = E[ Φ_Σ(x * sqrt(W/df)) ]

        参数:
            x: 分位数向量

        返回:
            多元t CDF
        """
        x = np.asarray(x, dtype=float)
        # 使用蒙特卡洛积分（种子固定保证可复现）
        rng = np.random.RandomState(42)
        n_mc = 10000
        w = rng.chisquare(self.df, size=n_mc) / self.df
        mvn = multivariate_normal(mean=np.zeros(self.dim), cov=self.corr_matrix)
        # 对每个样本计算多元正态CDF
        probs = np.array([
            mvn.cdf(x * np.sqrt(wi)) for wi in w
        ])
        return float(np.mean(probs))

    def cdf(self, u: np.ndarray) -> float:
        """
        计算Student-t Copula的CDF值。

        参数:
            u: 均匀分布分位数向量 (n,)

        返回:
            Copula CDF值
        """
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 1e-10, 1 - 1e-10)
        z = t_dist.ppf(u, df=self.df)  # 逆t分布
        return self._mv_t_cdf(z)

    def simulate(self, n_samples: int, random_state: int = 42) -> np.ndarray:
        """
        从Student-t Copula生成样本。

        参数:
            n_samples  : 样本数
            random_state: 随机种子

        返回:
            (n_samples, dim) 的均匀分布样本
        """
        rng = np.random.RandomState(random_state)
        z = rng.multivariate_normal(
            mean=np.zeros(self.dim), cov=self.corr_matrix, size=n_samples
        )
        w = rng.chisquare(self.df, size=n_samples) / self.df
        t_samples = z / np.sqrt(w)[:, np.newaxis]
        u = t_dist.cdf(t_samples, df=self.df)
        return u

    @classmethod
    def fit_from_data(cls, data: np.ndarray, df: float = 5.0) -> "StudentTCopula":
        """
        从数据拟合Student-t Copula。

        相关矩阵用Kendall's tau估计，自由度默认设为5（或可由MLE优化）。

        参数:
            data: (n_samples, n_features) 数据矩阵
            df  : 自由度，默认5.0

        返回:
            拟合后的StudentTCopula实例
        """
        data = np.asarray(data, dtype=float)
        tau_mat = kendall_tau_matrix(data)
        corr = np.sin(np.pi * tau_mat / 2.0)
        return cls(corr, df=df)


# ============================================================
#  Clayton Copula（双变量）
# ============================================================

class ClaytonCopula:
    """
    Clayton Copula（双变量）。

    C(u, v) = max((u^{-θ} + v^{-θ} - 1)^{-1/θ}, 0)

    Clayton Copula具有下尾相依性，适合信用风险建模。
    参数 θ > 0 表示正相关，θ越大相依越强。

    参数:
        theta: Clayton参数，θ > 0
    """

    def __init__(self, theta: float):
        if theta < 0:
            raise ValueError("Clayton Copula参数theta必须非负")
        self.theta = max(theta, 1e-8)

    def cdf(self, u: np.ndarray) -> float:
        """
        计算双变量Clayton Copula的CDF。

        C(u, v) = max((u^{-θ} + v^{-θ} - 1)^{-1/θ}, 0)

        参数:
            u: (u, v) 向量，每个元素在(0,1)

        返回:
            Copula CDF值
        """
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 1e-10, 1 - 1e-10)
        if len(u) != 2:
            raise ValueError("Clayton Copula目前仅支持双变量")

        u1, u2 = u[0], u[1]
        val = u1 ** (-self.theta) + u2 ** (-self.theta) - 1.0
        if val <= 0:
            return 0.0
        return float(val ** (-1.0 / self.theta))

    def simulate(self, n_samples: int, random_state: int = 42) -> np.ndarray:
        """
        从Clayton Copula生成样本（条件分布法）。

        参数:
            n_samples  : 样本数
            random_state: 随机种子

        返回:
            (n_samples, 2) 的均匀分布样本
        """
        rng = np.random.RandomState(random_state)
        # 生成均匀随机数
        s = rng.uniform(0, 1, n_samples)
        w = rng.uniform(0, 1, n_samples)

        # 条件分布法生成Clayton Copula样本
        # 给定 s, v = ((1 - s^{-θ} + s^{-θ} * w^{-θ/(θ+1)}) ) ...
        # Clayton条件逆: v = (1 + u1^{-θ} * (w^{-θ/(θ+1)} - 1))^{-1/θ}
        theta = self.theta
        u1 = s  # u1 = s
        # 逆条件CDF
        u2 = (1 + u1 ** (-theta) * (w ** (-theta / (theta + 1)) - 1)) ** (-1.0 / theta)
        u2 = np.clip(u2, 1e-10, 1 - 1e-10)

        return np.column_stack([u1, u2])

    @classmethod
    def fit_from_data(cls, x: np.ndarray, y: np.ndarray) -> "ClaytonCopula":
        """
        从双变量数据拟合Clayton Copula。

        使用Kendall's tau矩估计: θ = 2τ / (1 - τ)

        参数:
            x: 变量1数据
            y: 变量2数据

        返回:
            拟合后的ClaytonCopula实例
        """
        tau = kendall_tau(x, y)
        theta = clayton_theta_from_tau(tau)
        return cls(theta)

    @classmethod
    def fit_mle(cls, x: np.ndarray, y: np.ndarray) -> "ClaytonCopula":
        """
        使用最大似然估计拟合Clayton Copula。

        参数:
            x: 变量1数据
            y: 变量2数据

        返回:
            拟合后的ClaytonCopula实例
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        # 转为均匀分布（经验CDF）
        u1 = stats.rankdata(x) / (len(x) + 1)
        u2 = stats.rankdata(y) / (len(y) + 1)

        def neg_log_likelihood(theta):
            if theta < 0.01:
                return 1e10
            c = ClaytonCopula(theta)
            # Clayton Copula密度函数
            # c(u,v) = (1+θ) * (u*v)^{-θ-1} * (u^{-θ}+v^{-θ}-1)^{-1/θ-2}
            u1c = np.clip(u1, 1e-10, 1 - 1e-10)
            u2c = np.clip(u2, 1e-10, 1 - 1e-10)
            term = u1c ** (-theta) + u2c ** (-theta) - 1.0
            term = np.maximum(term, 1e-10)
            density = (1 + theta) * (u1c * u2c) ** (-theta - 1) * \
                      term ** (-1.0 / theta - 2)
            density = np.maximum(density, 1e-300)
            return -np.sum(np.log(density))

        result = minimize(neg_log_likelihood, x0=2.0, method="Nelder-Mead",
                          options={"xatol": 1e-6, "maxiter": 500})
        return cls(max(result.x[0], 1e-6))


# ============================================================
#  联合违约概率
# ============================================================

def joint_default_probability(
    copula,
    default_probs: np.ndarray,
) -> float:
    """
    计算联合违约概率。

    在Copula框架下，联合违约概率为：
        P(违约_1, ..., 违约_n) = C(PD_1, ..., PD_n)

    参数:
        copula       : Copula实例（GaussianCopula/StudentTCopula/ClaytonCopula）
        default_probs: 各公司的违约概率向量 (n,)

    返回:
        联合违约概率
    """
    return copula.cdf(default_probs)


def conditional_default_probability(
    copula,
    default_probs: np.ndarray,
    i: int,
    j: int,
) -> float:
    """
    计算条件违约概率 P(违约_i | 违约_j)。

    P(违约_i | 违约_j) = C(PD_i, PD_j) / PD_j

    参数:
        copula       : 双变量Copula实例
        default_probs: 违约概率向量
        i            : 公司i的索引
        j            : 公司j的索引

    返回:
        条件违约概率
    """
    pd_i = default_probs[i]
    pd_j = default_probs[j]

    if pd_j <= 0:
        return 0.0

    joint = copula.cdf([pd_i, pd_j])
    return joint / pd_j


# ============================================================
#  资产组合违约相关性分析
# ============================================================

def portfolio_default_correlation(
    panels: dict,
    kmv_results: pd.DataFrame,
    copula_type: str = "gaussian",
) -> dict:
    """
    资产组合违约相关性分析。

    使用KMV模型输出的违约概率（EDF）和股价数据，
    拟合Copula并计算联合违约概率。

    参数:
        panels      : 多公司面板数据
        kmv_results : KMV模型结果
        copula_type : "gaussian", "t", 或 "clayton"

    返回:
        dict，包含：
            - copula            : 拟合的Copula实例
            - corr_matrix       : 相关矩阵（Gaussian/t）
            - joint_pd          : 联合违约概率
            - marginal_pds      : 单个违约概率
            - tickers           : 股票代码列表
            - copula_type       : Copula类型
    """
    # 提取收益率数据用于拟合Copula
    tickers = list(panels.keys())
    n = len(tickers)

    if n < 2:
        raise ValueError("至少需要2家公司才能进行Copula分析")

    # 构建收益率矩阵（对齐日期）
    returns_dict = {}
    for ticker, panel in panels.items():
        prices = panel["price_data"]
        returns_dict[ticker] = prices.set_index("date")["log_return"]

    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna()

    if len(returns_df) < 30:
        raise ValueError("有效收益率数据不足30个交易日，无法拟合Copula")

    data = returns_df.values

    # 提取违约概率
    edf_map = dict(zip(kmv_results["ticker"], kmv_results["edf"]))
    pds = np.array([edf_map.get(t, 0.01) for t in tickers])

    # 拟合Copula
    if copula_type == "gaussian":
        copula = GaussianCopula.fit_from_data(data)
        corr_matrix = copula.corr_matrix
    elif copula_type == "t":
        copula = StudentTCopula.fit_from_data(data, df=5.0)
        corr_matrix = copula.corr_matrix
    elif copula_type == "clayton":
        # Clayton仅支持双变量，取前两家
        if n >= 2:
            copula = ClaytonCopula.fit_from_data(data[:, 0], data[:, 1])
            corr_matrix = np.array([[1.0, np.sin(np.pi * kendall_tau(data[:, 0], data[:, 1]) / 2)],
                                     [np.sin(np.pi * kendall_tau(data[:, 0], data[:, 1]) / 2), 1.0]])
        else:
            raise ValueError("Clayton Copula需要至少2个变量")
    else:
        raise ValueError(f"不支持的Copula类型: {copula_type}")

    # 计算联合违约概率
    if copula_type == "clayton":
        joint_pd = copula.cdf(pds[:2])
    else:
        joint_pd = copula.cdf(pds)

    return {
        "copula": copula,
        "corr_matrix": corr_matrix,
        "joint_pd": joint_pd,
        "marginal_pds": pds,
        "tickers": tickers,
        "copula_type": copula_type,
        "returns_data": returns_df,
    }


def copula_correlation_heatmap_data(corr_matrix: np.ndarray,
                                    tickers: list) -> pd.DataFrame:
    """
    生成Copula相关矩阵热力图数据。

    参数:
        corr_matrix: 相关系数矩阵
        tickers    : 股票代码列表

    返回:
        DataFrame，行列均为股票代码
    """
    return pd.DataFrame(corr_matrix, index=tickers, columns=tickers)
