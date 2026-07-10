# -*- coding: utf-8 -*-
"""
财务数据获取模块
=================

基于akshare接口获取A股上市公司真实财务数据，包括：
    - 资产负债表（总资产、总负债、流动资产/负债等）
    - 利润表（营业收入、净利润、财务费用等）
    - 股票市值与历史价格（用于计算股价波动率）
    - 信用分析面板数据构建

所有数据均来自akshare公开接口，不包含任何随机/伪造数据。
"""

import datetime
import numpy as np
import pandas as pd

# 尝试导入akshare
try:
    import akshare as ak
except ImportError:
    raise ImportError(
        "未安装akshare，请执行: pip install akshare>=1.12.0"
    )


def _normalize_ticker(ticker: str) -> str:
    """
    将用户输入的股票代码标准化为6位数字格式（不含交易所前缀）。

    支持的输入格式：
        - "600519" / "000858" / "601318"
        - "sh600519" / "sz000858"
        - "SH600519" / "SZ000858"

    参数:
        ticker: 股票代码字符串

    返回:
        6位数字股票代码字符串，如 "600519"
    """
    t = ticker.strip().upper().replace("S", "")  # 去掉sh/sz前缀
    # 如果去除前缀后仍以数字开头，取数字部分
    digits = ""
    for ch in t:
        if ch.isdigit():
            digits += ch
    if len(digits) == 6:
        return digits
    # 兜底：如果无法解析出6位数字，原样返回
    return ticker.strip()


def _get_exchange_prefix(ticker: str) -> str:
    """
    根据股票代码判断交易所前缀（SH/SZ）。

    沪市: 6开头（600/601/603/605/688）
    深市: 0开头（000/001/002/003/300/301）

    参数:
        ticker: 6位数字股票代码

    返回:
        交易所前缀 "SH" 或 "SZ"
    """
    code = _normalize_ticker(ticker)
    if code.startswith("6"):
        return "SH"
    else:
        return "SZ"


def _format_em_symbol(ticker: str) -> str:
    """
    转换为akshare EM接口所需的带前缀格式（如 SH600519）。

    参数:
        ticker: 股票代码

    返回:
        带交易所前缀的符号，如 "SH600519"
    """
    code = _normalize_ticker(ticker)
    prefix = _get_exchange_prefix(code)
    return f"{prefix}{code}"


# ============================================================
#  资产负债表获取
# ============================================================

def fetch_balance_sheet(ticker: str, n_reports: int = 8) -> pd.DataFrame:
    """
    获取上市公司资产负债表数据（最近n_reports期报告）。

    使用 akshare 的 stock_balance_sheet_by_report_em 接口，
    获取按报告期排列的资产负债表。

    参数:
        ticker: 股票代码，如 "600519"
        n_reports: 获取最近几期报告，默认8期

    返回:
        DataFrame，包含以下列（已标准化命名）：
            - report_date      : 报告期
            - total_assets     : 资产总计
            - total_liab       : 负债合计
            - current_assets   : 流动资产合计
            - current_liab     : 流动负债合计
            - total_equity     : 股东权益合计
            - long_term_liab   : 长期负债合计（计算）
            - retained_earnings: 留存收益（未分配利润+盈余公积）
            - inventory        : 存货
            - paid_in_capital  : 实收资本（股本）
    """
    em_symbol = _format_em_symbol(ticker)

    # 获取资产负债表
    df = ak.stock_balance_sheet_by_report_em(symbol=em_symbol)

    if df is None or len(df) == 0:
        raise ValueError(f"无法获取 {ticker} 的资产负债表数据，请检查股票代码")

    # 取最近n_reports期
    df = df.head(n_reports).copy()

    # 标准化列名（akshare的EM接口可能使用不同列名，这里做兼容）
    col_map = {
        "REPORT_DATE_NAME": "report_date",
        "TOTAL_ASSETS": "total_assets",
        "TOTAL_LIABILITIES": "total_liab",
        "TOTAL_CURRENT_ASSETS": "current_assets",
        "TOTAL_CURRENT_LIABILITIES": "current_liab",
        "TOTAL_EQUITY": "total_equity",
        "UNDISTRIBUTED_PROFIT": "undistributed_profit",
        "SURPLUS_RESERVE": "surplus_reserve",
        "INVENTORY": "inventory",
        "TOTAL_PAID_IN_CAPITAL": "paid_in_capital",
    }

    result = pd.DataFrame()
    for ak_col, std_col in col_map.items():
        if ak_col in df.columns:
            result[std_col] = df[ak_col].values
        else:
            result[std_col] = np.nan

    # 计算长期负债 = 总负债 - 流动负债
    result["long_term_liab"] = result["total_liab"] - result["current_liab"]

    # 计算留存收益 = 未分配利润 + 盈余公积
    result["retained_earnings"] = result["undistributed_profit"].fillna(0) + \
                                  result["surplus_reserve"].fillna(0)
    # 如果留存收益仍为0，用股东权益 - 实收资本近似
    mask = (result["retained_earnings"].fillna(0) == 0) & \
           result["total_equity"].notna() & result["paid_in_capital"].notna()
    result.loc[mask, "retained_earnings"] = \
        result.loc[mask, "total_equity"] - result.loc[mask, "paid_in_capital"]

    # 数值列转换为float
    num_cols = ["total_assets", "total_liab", "current_assets",
                "current_liab", "total_equity", "long_term_liab",
                "retained_earnings", "inventory", "paid_in_capital"]
    for col in num_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 按报告期排序（旧的在前）
    result = result.sort_values("report_date").reset_index(drop=True)

    return result


# ============================================================
#  利润表获取
# ============================================================

def fetch_income_statement(ticker: str, n_reports: int = 8) -> pd.DataFrame:
    """
    获取上市公司利润表数据（最近n_reports期报告）。

    使用 akshare 的 stock_profit_sheet_by_report_em 接口。

    参数:
        ticker: 股票代码，如 "600519"
        n_reports: 获取最近几期报告，默认8期

    返回:
        DataFrame，包含以下列：
            - report_date       : 报告期
            - revenue           : 营业收入
            - net_profit        : 净利润
            - total_profit      : 利润总额
            - finance_expense   : 财务费用（利息支出代理）
            - operate_profit    : 营业利润
            - income_tax        : 所得税费用
    """
    em_symbol = _format_em_symbol(ticker)

    df = ak.stock_profit_sheet_by_report_em(symbol=em_symbol)

    if df is None or len(df) == 0:
        raise ValueError(f"无法获取 {ticker} 的利润表数据，请检查股票代码")

    df = df.head(n_reports).copy()

    col_map = {
        "REPORT_DATE_NAME": "report_date",
        "OPERATE_INCOME": "revenue",
        "NETPROFIT": "net_profit",
        "TOTAL_PROFIT": "total_profit",
        "FINANCE_EXPENSE": "finance_expense",
        "OPERATE_PROFIT": "operate_profit",
        "INCOME_TAX": "income_tax",
    }

    result = pd.DataFrame()
    for ak_col, std_col in col_map.items():
        if ak_col in df.columns:
            result[std_col] = df[ak_col].values
        else:
            result[std_col] = np.nan

    # 数值列转换
    num_cols = ["revenue", "net_profit", "total_profit",
                "finance_expense", "operate_profit", "income_tax"]
    for col in num_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result = result.sort_values("report_date").reset_index(drop=True)

    return result


# ============================================================
#  股票市值与价格数据获取
# ============================================================

def fetch_stock_price(ticker: str, n_days: int = 252) -> pd.DataFrame:
    """
    获取股票历史日线行情，用于计算股价波动率。

    使用 akshare 的 stock_zh_a_hist 接口（前复权）。

    参数:
        ticker: 股票代码，如 "600519"
        n_days: 获取最近多少个交易日，默认252（约一年）

    返回:
        DataFrame，包含：
            - date   : 日期
            - close  : 收盘价（前复权）
            - log_return : 对数收益率
    """
    code = _normalize_ticker(ticker)

    # 计算日期范围（多取些天数以确保有足够交易日）
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=int(n_days * 1.6) + 30)

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        adjust="qfq",  # 前复权
    )

    if df is None or len(df) == 0:
        raise ValueError(f"无法获取 {ticker} 的历史行情数据")

    # 标准化列名
    df = df.rename(columns={
        "日期": "date",
        "收盘": "close",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
    })

    # 计算对数收益率
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)

    # 取最近n_days个交易日
    if len(df) > n_days:
        df = df.tail(n_days).reset_index(drop=True)

    return df[["date", "close", "log_return"]]


def fetch_market_cap(ticker: str) -> dict:
    """
    获取上市公司市值信息。

    使用 akshare 的 stock_individual_info_em 接口。

    参数:
        ticker: 股票代码，如 "600519"

    返回:
        dict，包含：
            - market_cap    : 总市值（元）
            - circulating_market_cap : 流通市值（元）
            - industry      : 所属行业
            - name          : 股票简称
            - list_date     : 上市日期
    """
    code = _normalize_ticker(ticker)

    df = ak.stock_individual_info_em(symbol=code)

    if df is None or len(df) == 0:
        raise ValueError(f"无法获取 {ticker} 的个股信息")

    # stock_individual_info_em 返回 item/value 两列的DataFrame
    info = {}
    for _, row in df.iterrows():
        item = str(row["item"]).strip()
        value = row["value"]
        info[item] = value

    result = {
        "market_cap": _parse_float(info.get("总市值", np.nan)),
        "circulating_market_cap": _parse_float(info.get("流通市值", np.nan)),
        "industry": str(info.get("行业", "未知")),
        "name": str(info.get("股票简称", ticker)),
        "list_date": str(info.get("上市时间", "")),
    }

    return result


def _parse_float(val) -> float:
    """将字符串格式的数值转为float（处理'亿'、'万'等单位）。"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    try:
        # 尝试直接转换
        return float(s)
    except ValueError:
        # 处理带单位的字符串
        if "亿" in s:
            s = s.replace("亿", "")
            return float(s) * 1e8
        elif "万" in s:
            s = s.replace("万", "")
            return float(s) * 1e4
        elif "万亿" in s:
            s = s.replace("万亿", "")
            return float(s) * 1e12
        return np.nan


# ============================================================
#  信用分析面板数据构建
# ============================================================

def build_credit_panel(ticker: str) -> dict:
    """
    构建单个公司的信用分析面板数据。

    汇总资产负债表、利润表、市值与价格数据，
    形成用于信用风险分析的完整数据集。

    参数:
        ticker: 股票代码，如 "600519"

    返回:
        dict，包含：
            - ticker          : 股票代码
            - name            : 股票简称
            - industry        : 所属行业
            - balance_sheet   : 资产负债表DataFrame
            - income_stmt     : 利润表DataFrame
            - price_data      : 价格数据DataFrame
            - market_cap      : 总市值（元）
            - latest_balance  : 最新一期资产负债表（Series）
            - latest_income   : 最新一期利润表（Series）
            - equity_value    : 股权价值（=总市值，元）
            - stock_volatility: 股价年化波动率
            - total_liab      : 总负债（元）
            - short_term_liab : 短期负债（流动负债，元）
            - long_term_liab  : 长期负债（元）
            - default_point   : 违约点（短期负债+0.5*长期负债）
    """
    # 获取各类数据
    balance = fetch_balance_sheet(ticker, n_reports=8)
    income = fetch_income_statement(ticker, n_reports=8)
    prices = fetch_stock_price(ticker, n_days=252)
    mkt = fetch_market_cap(ticker)

    # 最新一期财务数据
    latest_balance = balance.iloc[-1]
    latest_income = income.iloc[-1]

    # 股权价值 = 总市值
    equity_value = mkt["market_cap"]
    if np.isnan(equity_value) or equity_value <= 0:
        # 如果市值获取失败，用股价 * 总股本估算
        # 但这里我们直接抛出错误，因为我们需要真实市值
        raise ValueError(f"无法获取 {ticker} 的有效市值数据")

    # 股价年化波动率 = 日收益率标准差 * sqrt(252)
    stock_vol = prices["log_return"].std() * np.sqrt(252)
    if np.isnan(stock_vol) or stock_vol <= 0:
        raise ValueError(f"无法计算 {ticker} 的股价波动率")

    # 财务数据提取
    total_liab = float(latest_balance.get("total_liab", 0))
    current_liab = float(latest_balance.get("current_liab", 0))
    long_term_liab = float(latest_balance.get("long_term_liab", 0))
    total_assets = float(latest_balance.get("total_assets", 0))

    # 违约点（Default Point）
    # KMV模型中：违约点 = 短期负债 + 0.5 * 长期负债
    # 短期负债 ≈ 流动负债，长期负债 = 总负债 - 流动负债
    default_point = current_liab + 0.5 * long_term_liab

    if default_point <= 0:
        # 如果违约点为0或负，使用总负债
        default_point = total_liab

    panel = {
        "ticker": _normalize_ticker(ticker),
        "name": mkt["name"],
        "industry": mkt["industry"],
        "balance_sheet": balance,
        "income_stmt": income,
        "price_data": prices,
        "market_cap": equity_value,
        "latest_balance": latest_balance,
        "latest_income": latest_income,
        "equity_value": equity_value,
        "stock_volatility": stock_vol,
        "total_liab": total_liab,
        "short_term_liab": current_liab,
        "long_term_liab": long_term_liab,
        "total_assets": total_assets,
        "default_point": default_point,
    }

    return panel


def build_multi_company_panel(tickers: list) -> dict:
    """
    构建多家公司的信用分析面板。

    参数:
        tickers: 股票代码列表，如 ["600519", "000858", "601318"]

    返回:
        dict，键为股票代码，值为build_credit_panel的返回值
    """
    panels = {}
    for ticker in tickers:
        try:
            panel = build_credit_panel(ticker)
            panels[panel["ticker"]] = panel
        except Exception as e:
            print(f"[警告] 获取 {ticker} 数据失败: {e}")
    return panels


# ============================================================
#  指数成分股获取
# ============================================================

def get_index_constituents(index: str, top_n: int = 20) -> list:
    """
    获取指数成分股列表。

    参数:
        index: 指数代码，支持 "hs300"（沪深300）、"zz500"（中证500）
        top_n: 取前n只成分股

    返回:
        股票代码列表
    """
    index = index.lower().strip()

    if index in ("hs300", "沪深300", "000300"):
        df = ak.index_stock_cons_csindex(symbol="000300")
    elif index in ("zz500", "中证500", "000905"):
        df = ak.index_stock_cons_csindex(symbol="000905")
    elif index in ("sz50", "上证50", "000016"):
        df = ak.index_stock_cons_csindex(symbol="000016")
    else:
        raise ValueError(f"不支持的指数: {index}，支持: hs300, zz500, sz50")

    if df is None or len(df) == 0:
        raise ValueError(f"无法获取指数 {index} 的成分股列表")

    # 成分股代码列
    code_col = "成分券代码" if "成分券代码" in df.columns else df.columns[0]
    codes = df[code_col].astype(str).tolist()

    # 去重并取前top_n
    codes = list(dict.fromkeys(codes))[:top_n]

    return codes


# ============================================================
#  辅助：财务比率计算
# ============================================================

def compute_financial_ratios(panel: dict) -> dict:
    """
    基于信用分析面板数据计算关键财务比率。

    参数:
        panel: build_credit_panel 返回的面板数据

    返回:
        dict，包含各财务比率：
            - current_ratio      : 流动比率 = 流动资产 / 流动负债
            - quick_ratio         : 速动比率 = (流动资产 - 存货) / 流动负债
            - debt_to_assets      : 资产负债率 = 总负债 / 总资产
            - debt_to_equity      : 产权比率 = 总负债 / 股东权益
            - equity_multiplier   : 权益乘数 = 总资产 / 股东权益
            - interest_coverage   : 利息保障倍数 = (利润总额+财务费用) / 财务费用
            - roa                 : 总资产收益率 = 净利润 / 总资产
            - roe                 : 净资产收益率 = 净利润 / 股东权益
    """
    lb = panel["latest_balance"]
    li = panel["latest_income"]

    current_assets = float(lb.get("current_assets", 0))
    current_liab = float(lb.get("current_liab", 0))
    total_assets = float(lb.get("total_assets", 0))
    total_liab = float(lb.get("total_liab", 0))
    total_equity = float(lb.get("total_equity", 0))

    revenue = float(li.get("revenue", 0))
    net_profit = float(li.get("net_profit", 0))
    finance_expense = abs(float(li.get("finance_expense", 0)))
    total_profit = float(li.get("total_profit", 0))

    ratios = {
        "current_ratio": current_assets / current_liab if current_liab > 0 else np.nan,
        "debt_to_assets": total_liab / total_assets if total_assets > 0 else np.nan,
        "debt_to_equity": total_liab / total_equity if total_equity > 0 else np.nan,
        "equity_multiplier": total_assets / total_equity if total_equity > 0 else np.nan,
        "interest_coverage": (total_profit + finance_expense) / finance_expense if finance_expense > 0 else np.nan,
        "roa": net_profit / total_assets if total_assets > 0 else np.nan,
        "roe": net_profit / total_equity if total_equity > 0 else np.nan,
        "net_margin": net_profit / revenue if revenue > 0 else np.nan,
    }

    return ratios
