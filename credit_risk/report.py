# -*- coding: utf-8 -*-
"""
HTML报告与可视化图表生成模块
================================

本模块使用matplotlib生成信用风险可视化图表，并构建完整的HTML报告。

生成的图表包括：
    - 违约距离排名图（横向条形图）
    - 资产价值vs违约点图（散点图）
    - 信用评分分布图（条形图）
    - Copula依赖结构热力图
    - 组合损失分布直方图

HTML报告通过base64嵌入所有图表图片，生成独立可分享的HTML文件。
"""

import os
import base64
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt

# 设置中文字体（尝试多个常见中文字体）
plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
    "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


# ============================================================
#  图表生成函数
# ============================================================

def plot_dd_ranking(kmv_results: pd.DataFrame, save_path: str = None) -> str:
    """
    生成违约距离排名图（横向条形图）。

    按违约距离从小到大排列，DD越小风险越大（红色越深）。

    参数:
        kmv_results: KMV模型结果DataFrame
        save_path  : 图片保存路径，None则不保存

    返回:
        图片文件路径（若保存），否则返回None
    """
    fig, ax = plt.subplots(figsize=(10, max(4, len(kmv_results) * 0.5)))

    df = kmv_results.sort_values("dd", ascending=True)
    names = df["name"].values
    dds = df["dd"].values

    # 颜色映射：DD越小越红
    colors = plt.cm.RdYlGn(np.clip((dds - dds.min()) / (dds.max() - dds.min() + 1e-8), 0, 1))

    bars = ax.barh(range(len(names)), dds, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("违约距离 (Distance to Default)", fontsize=12)
    ax.set_title("违约距离排名（DD越小 = 风险越大）", fontsize=14, fontweight="bold")

    # 在条形上标注DD值
    for i, (bar, dd) in enumerate(zip(bars, dds)):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{dd:.2f}", va="center", fontsize=9)

    ax.axvline(x=0, color="black", linewidth=0.5)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    path = save_path
    plt.close(fig)
    return path


def plot_asset_vs_default_point(kmv_results: pd.DataFrame,
                                 save_path: str = None) -> str:
    """
    生成资产价值vs违约点图（散点图）。

    横轴为违约点，纵轴为资产价值。45度线以上为安全区。

    参数:
        kmv_results: KMV模型结果DataFrame
        save_path  : 图片保存路径

    返回:
        图片文件路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    av = kmv_results["asset_value"].values / 1e8  # 转为亿元
    dp = kmv_results["default_point"].values / 1e8
    names = kmv_results["name"].values

    # 散点图
    scatter = ax.scatter(dp, av, c=kmv_results["dd"].values,
                          cmap="RdYlGn", s=150, edgecolors="black", linewidth=0.5,
                          zorder=5)
    plt.colorbar(scatter, label="违约距离 DD")

    # 45度线
    all_vals = np.concatenate([dp, av])
    max_val = np.max(all_vals) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="资产价值=违约点", alpha=0.5)

    # 标注公司名称
    for i, name in enumerate(names):
        ax.annotate(name, (dp[i], av[i]),
                     textcoords="offset points", xytext=(8, 5), fontsize=8)

    ax.set_xlabel("违约点（亿元）", fontsize=12)
    ax.set_ylabel("资产价值（亿元）", fontsize=12)
    ax.set_title("资产价值 vs 违约点", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    path = save_path
    plt.close(fig)
    return path


def plot_credit_score_distribution(scoring_results: pd.DataFrame,
                                    save_path: str = None) -> str:
    """
    生成信用评分分布图（条形图）。

    显示各公司的综合信用评分及评级。

    参数:
        scoring_results: 信用评分结果DataFrame
        save_path      : 图片保存路径

    返回:
        图片文件路径
    """
    fig, ax = plt.subplots(figsize=(10, max(4, len(scoring_results) * 0.5)))

    df = scoring_results.sort_values("comprehensive_score", ascending=True)
    names = df["name"].values
    scores = df["comprehensive_score"].values
    ratings = df["rating"].values

    # 颜色映射
    colors = plt.cm.RdYlGn(np.clip(scores / 100, 0, 1))

    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("综合信用评分", fontsize=12)
    ax.set_title("信用评分分布", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)

    # 标注评级
    for i, (bar, score, rating) in enumerate(zip(bars, scores, ratings)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f} [{rating}]", va="center", fontsize=9)

    # 评级分界线
    for threshold, label in [(35, "B/CCC"), (55, "BB/BBB"), (75, "A/AA")]:
        ax.axvline(x=threshold, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    path = save_path
    plt.close(fig)
    return path


def plot_copula_heatmap(corr_matrix: np.ndarray, tickers: list,
                         save_path: str = None) -> str:
    """
    生成Copula依赖结构热力图。

    参数:
        corr_matrix: 相关系数矩阵
        tickers    : 股票代码列表（用于标签）
        save_path  : 图片保存路径

    返回:
        图片文件路径
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    corr = np.asarray(corr_matrix)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # 设置刻度
    labels = [t[:6] for t in tickers]  # 截断代码长度
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # 添加数值标注
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="相关系数")
    ax.set_title("Copula依赖结构热力图", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    path = save_path
    plt.close(fig)
    return path


def plot_loss_distribution(losses: np.ndarray, var_total: float,
                            el: float, confidence: float,
                            save_path: str = None) -> str:
    """
    生成组合损失分布直方图。

    标注预期损失(EL)和VaR。

    参数:
        losses    : 损失分布数组
        var_total : 总VaR值
        el        : 预期损失
        confidence: 置信水平
        save_path : 图片保存路径

    返回:
        图片文件路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 直方图
    n_bins = min(100, max(20, len(losses) // 500))
    ax.hist(losses, bins=n_bins, density=True, color="steelblue",
             edgecolor="white", linewidth=0.5, alpha=0.7)

    # 标注EL
    ax.axvline(x=el, color="green", linewidth=2, linestyle="-",
               label=f"预期损失 EL = {el/1e8:.2f}亿")

    # 标注VaR
    ax.axvline(x=var_total, color="red", linewidth=2, linestyle="--",
               label=f"VaR({confidence*100:.1f}%) = {var_total/1e8:.2f}亿")

    ax.set_xlabel("组合信用损失（元）", fontsize=12)
    ax.set_ylabel("概率密度", fontsize=12)
    ax.set_title("蒙特卡洛模拟组合信用损失分布", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    # 格式化x轴为亿元
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e8:.0f}"))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    path = save_path
    plt.close(fig)
    return path


# ============================================================
#  图片转Base64
# ============================================================

def image_to_base64(image_path: str) -> str:
    """
    将图片文件转换为base64编码字符串。

    参数:
        image_path: 图片文件路径

    返回:
        base64编码的图片数据URI
    """
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{img_data}"


# ============================================================
#  信用风险预警表
# ============================================================

def generate_warning_table(kmv_results: pd.DataFrame,
                           scoring_results: pd.DataFrame) -> pd.DataFrame:
    """
    生成信用风险预警表。

    整合KMV模型和信用评分结果，对高风险公司进行预警标记。

    参数:
        kmv_results    : KMV模型结果
        scoring_results: 信用评分结果

    返回:
        DataFrame，预警表
    """
    # 合并KMV和评分结果
    kmv_df = kmv_results[["ticker", "name", "industry", "dd", "edf"]].copy()
    kmv_df.columns = ["ticker", "name", "industry", "dd", "edf"]

    score_df = scoring_results[["ticker", "altman_z", "altman_zone",
                                  "altman_rating", "comprehensive_score", "rating"]].copy()

    merged = kmv_df.merge(score_df, on="ticker", how="outer")

    # 预警等级
    def _warning_level(row):
        dd = row.get("dd", np.nan)
        edf = row.get("edf", np.nan)
        zone = row.get("altman_zone", "")
        score = row.get("comprehensive_score", np.nan)

        if np.isnan(score) and np.isnan(dd):
            return "数据不足"
        if (not np.isnan(dd) and dd < 1.0) or (not np.isnan(edf) and edf > 0.10) or \
           zone == "危险区" or (not np.isnan(score) and score < 35):
            return "高风险"
        elif (not np.isnan(dd) and dd < 2.0) or (not np.isnan(edf) and edf > 0.02) or \
             zone == "灰色区" or (not np.isnan(score) and score < 55):
            return "中风险"
        else:
            return "低风险"

    merged["warning_level"] = merged.apply(_warning_level, axis=1)

    # 格式化
    merged["dd"] = merged["dd"].round(3)
    merged["edf"] = (merged["edf"] * 100).round(4)  # 转为百分比
    merged["altman_z"] = merged["altman_z"].round(3)
    merged["comprehensive_score"] = merged["comprehensive_score"].round(1)

    merged = merged.sort_values("warning_level").reset_index(drop=True)
    return merged


# ============================================================
#  HTML报告生成
# ============================================================

def generate_html_report(
    kmv_results: pd.DataFrame,
    scoring_results: pd.DataFrame,
    copula_result: dict = None,
    var_result: dict = None,
    industry_stats: pd.DataFrame = None,
    output_dir: str = "output",
    title: str = "信用风险评估报告",
) -> str:
    """
    生成完整的HTML信用风险评估报告。

    报告包含：
        - 报告概览与生成时间
        - KMV模型结果（违约距离、违约概率排名）
        - 信用评分结果（Z-Score、综合评分）
        - Copula依赖结构分析
        - 组合信用VaR与经济资本
        - 信用风险预警表
        - 所有可视化图表（base64嵌入）

    参数:
        kmv_results    : KMV模型结果DataFrame
        scoring_results: 信用评分结果DataFrame
        copula_result  : Copula分析结果
        var_result     : 组合VaR评估结果
        industry_stats : 行业统计
        output_dir     : 输出目录
        title          : 报告标题

    返回:
        HTML报告文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 图片保存路径
    chart_dir = os.path.join(output_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)

    images = {}

    # 生成图表
    print("  生成违约距离排名图...")
    path = os.path.join(chart_dir, "dd_ranking.png")
    plot_dd_ranking(kmv_results, save_path=path)
    images["dd_ranking"] = image_to_base64(path)

    print("  生成资产价值vs违约点图...")
    path = os.path.join(chart_dir, "asset_vs_dp.png")
    plot_asset_vs_default_point(kmv_results, save_path=path)
    images["asset_vs_dp"] = image_to_base64(path)

    print("  生成信用评分分布图...")
    path = os.path.join(chart_dir, "score_dist.png")
    plot_credit_score_distribution(scoring_results, save_path=path)
    images["score_dist"] = image_to_base64(path)

    if copula_result is not None:
        print("  生成Copula热力图...")
        path = os.path.join(chart_dir, "copula_heatmap.png")
        plot_copula_heatmap(copula_result["corr_matrix"],
                            copula_result["tickers"], save_path=path)
        images["copula_heatmap"] = image_to_base64(path)

    if var_result is not None:
        print("  生成损失分布图...")
        mc = var_result["mc_result"]
        path = os.path.join(chart_dir, "loss_dist.png")
        plot_loss_distribution(mc["losses"], mc["var_total"],
                                mc["EL_sim"], mc["confidence"], save_path=path)
        images["loss_dist"] = image_to_base64(path)

    # 生成预警表
    print("  生成信用风险预警表...")
    warning_table = generate_warning_table(kmv_results, scoring_results)

    # 构建HTML
    html = _build_html(
        title=title,
        kmv_results=kmv_results,
        scoring_results=scoring_results,
        copula_result=copula_result,
        var_result=var_result,
        industry_stats=industry_stats,
        warning_table=warning_table,
        images=images,
    )

    # 保存HTML
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"credit_risk_report_{timestamp}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path


def _build_html(title, kmv_results, scoring_results, copula_result,
                var_result, industry_stats, warning_table, images) -> str:
    """
    构建HTML报告内容。

    参数:
        各分析结果数据

    返回:
        完整HTML字符串
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_companies = len(kmv_results)

    # KMV结果表格HTML
    kmv_table_html = _df_to_html(kmv_results[["ticker", "name", "industry",
                                              "asset_value", "default_point",
                                              "dd", "edf", "leverage_ratio"]])

    # 评分结果表格HTML
    score_cols = ["ticker", "name", "industry", "altman_z", "altman_zone",
                  "altman_rating", "comprehensive_score", "rating"]
    score_table_html = _df_to_html(scoring_results[[c for c in score_cols
                                                       if c in scoring_results.columns]])

    # 预警表HTML
    warning_html = _df_to_html(warning_table)

    # 行业统计
    industry_html = ""
    if industry_stats is not None and len(industry_stats) > 0:
        industry_html = _df_to_html(industry_stats)

    # 组合VaR摘要
    var_html = ""
    if var_result is not None:
        ps = var_result["portfolio_summary"]
        va = var_result["var_analytic"]
        mc = var_result["mc_result"]
        var_html = f"""
        <div class="card">
            <h2>组合信用VaR与经济资本</h2>
            <table class="summary-table">
                <tr><td>资产数量</td><td>{ps['n_assets']}</td></tr>
                <tr><td>总风险暴露 (EAD)</td><td>{ps['total_ead']/1e8:.2f} 亿元</td></tr>
                <tr><td>预期损失 (EL)</td><td>{ps['total_el']/1e8:.4f} 亿元</td></tr>
                <tr><td>意外损失 (UL)</td><td>{ps['total_ul']/1e8:.4f} 亿元</td></tr>
                <tr><td>信用VaR (解析法)</td><td>{ps['credit_var']/1e8:.4f} 亿元</td></tr>
                <tr><td>经济资本 (EC)</td><td>{ps['ec']/1e8:.4f} 亿元</td></tr>
                <tr><td colspan="2" style="background:#e8e8e8;"><b>蒙特卡洛模拟 ({mc['n_simulations']:,}次)</b></td></tr>
                <tr><td>模拟预期损失</td><td>{ps['mc_el']/1e8:.4f} 亿元</td></tr>
                <tr><td>模拟意外损失</td><td>{ps['mc_ul']/1e8:.4f} 亿元</td></tr>
                <tr><td>模拟信用VaR ({mc['confidence']*100:.1f}%)</td><td>{ps['mc_credit_var']/1e8:.4f} 亿元</td></tr>
                <tr><td>最坏情景损失</td><td>{ps['mc_worst_loss']/1e8:.4f} 亿元</td></tr>
            </table>
        </div>
        """

    # Copula分析
    copula_html = ""
    if copula_result is not None:
        copula_html = f"""
        <div class="card">
            <h2>Copula依赖结构分析 ({copula_result['copula_type'].upper()})</h2>
            <p>联合违约概率: <b>{copula_result['joint_pd']:.6f}</b> ({copula_result['joint_pd']*100:.4f}%)</p>
            <img src="{images.get('copula_heatmap', '')}" style="max-width:600px;width:100%;" />
        </div>
        """

    # 风险贡献表
    contrib_html = ""
    if var_result is not None:
        contrib = var_result["contributions"]
        contrib_html = f"""
        <div class="card">
            <h2>边际风险贡献</h2>
            {_df_to_html(contrib)}
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif;
            margin: 0; padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #1a5276;
            text-align: center;
            border-bottom: 3px solid #1a5276;
            padding-bottom: 10px;
        }}
        .header-info {{
            text-align: center;
            color: #666;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #1a5276;
            border-left: 4px solid #1a5276;
            padding-left: 10px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
            font-size: 13px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px 10px;
            text-align: center;
        }}
        th {{
            background-color: #1a5276;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .summary-table {{
            width: 60%;
            margin: 10px auto;
        }}
        .summary-table td:first-child {{
            text-align: right;
            font-weight: bold;
            width: 50%;
        }}
        .warning-high {{ background-color: #ffcccc !important; }}
        .warning-medium {{ background-color: #fff3cc !important; }}
        .warning-low {{ background-color: #ccffcc !important; }}
        img {{
            display: block;
            margin: 15px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .footer {{
            text-align: center;
            color: #999;
            margin-top: 30px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="header-info">
        <p>生成时间: {now} | 评估公司数: {n_companies} 家</p>
    </div>

    <div class="card">
        <h2>KMV-Merton结构化违约模型结果</h2>
        <p>违约距离(DD)越大，违约风险越小。违约概率(EDF) = N(-DD)。</p>
        <img src="{images['dd_ranking']}" style="max-width:700px;width:100%;" />
        <img src="{images['asset_vs_dp']}" style="max-width:600px;width:100%;" />
        {kmv_table_html}
    </div>

    <div class="card">
        <h2>信用评分结果</h2>
        <p>Altman Z-Score: Z&gt;2.99安全区, 1.81~2.99灰色区, Z&lt;1.81危险区。</p>
        <img src="{images['score_dist']}" style="max-width:700px;width:100%;" />
        {score_table_html}
    </div>

    {copula_html}

    {var_html}

    <div class="card">
        <h2>信用风险预警表</h2>
        {warning_html}
    </div>

    {contrib_html}

    <div class="card">
        <h2>行业基准比较</h2>
        {industry_html}
    </div>

    {f'''<div class="card">
        <h2>组合信用损失分布</h2>
        <img src="{images.get('loss_dist', '')}" style="max-width:700px;width:100%;" />
    </div>''' if 'loss_dist' in images else ''}

    <div class="footer">
        <p>本报告由信用风险建模工具(Credit Risk Modeling Toolkit)自动生成</p>
        <p>数据来源: akshare (A股真实财务数据) | 报告仅供参考，不构成投资建议</p>
    </div>
</body>
</html>"""

    return html


def _df_to_html(df: pd.DataFrame) -> str:
    """
    将DataFrame转换为格式化的HTML表格（含风险着色）。

    参数:
        df: 数据表

    返回:
        HTML表格字符串
    """
    # 格式化数值列
    display_df = df.copy()
    for col in display_df.columns:
        if display_df[col].dtype in ["float64", "float32"]:
            if "edf" in col.lower():
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x*100:.4f}%" if pd.notna(x) else "-"
                )
            elif "asset_value" in col or "default_point" in col or "leverage" in col.lower():
                if "leverage" in col.lower():
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "-"
                    )
                else:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x/1e8:.2f}亿" if pd.notna(x) else "-"
                    )
            elif "dd" in col.lower():
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "-"
                )
            elif "score" in col.lower():
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else "-"
                )
            else:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "-"
                )

    html = display_df.to_html(index=False, escape=False, classes="data-table", border=0)

    # 预警着色
    if "warning_level" in display_df.columns:
        html = html.replace("<tr>", "<tr>")
        # 简单着色逻辑
        for idx, row in display_df.iterrows():
            level = row.get("warning_level", "")
            if level == "高风险":
                html = html.replace(
                    display_df.iloc[idx:idx+1].to_html(index=False, header=False, border=0),
                    display_df.iloc[idx:idx+1].to_html(index=False, header=False, border=0,
                                                         classes="warning-high")
                )

    return html
