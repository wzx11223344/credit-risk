#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
信用风险建模工具 - CLI入口
============================

命令行用法:
    # 指定股票代码评估
    python assess.py --tickers 600519,000858,601318

    # 使用指数成分股评估
    python assess.py --index hs300 --top 10

    # 指定分析方法
    python assess.py --tickers 600519,000858 --method kmv
    python assess.py --tickers 600519,000858 --method copula
    python assess.py --tickers 600519,000858 --method var
    python assess.py --tickers 600519,000858 --method scoring
    python assess.py --tickers 600519,000858 --method all

    # 完整评估（默认）
    python assess.py --tickers 600519,000858,601318 --method all

参数说明:
    --tickers : 逗号分隔的股票代码列表
    --index   : 指数代码（hs300/zz500/sz50）
    --top     : 使用指数时取前N只成分股
    --method  : 分析方法（kmv/copula/var/scoring/all）
    --output  : 输出目录，默认 output/
    --rf      : 无风险利率（小数），默认自动获取
    --sim     : 蒙特卡洛模拟次数，默认100000
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入项目模块
from credit_risk import data, kmv_model, copula as copula_mod, var as var_mod
from credit_risk import scoring as scoring_mod, report as report_mod

# rich 进度显示
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table as RichTable
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if RICH_AVAILABLE:
    console = Console()
else:
    console = None


def _print(msg: str, style: str = ""):
    """统一输出函数，兼容rich和普通print。"""
    if RICH_AVAILABLE:
        console.print(msg, style=style)
    else:
        print(msg)


def _progress_context():
    """创建进度上下文，兼容rich和普通模式。"""
    if RICH_AVAILABLE:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        )
    else:
        return None


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="信用风险建模工具 - 专业级信用风险评估系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python assess.py --tickers 600519,000858,601318
    python assess.py --index hs300 --top 10 --method all
    python assess.py --tickers 600519 --method kmv
        """,
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="逗号分隔的股票代码列表，如 600519,000858,601318",
    )
    parser.add_argument(
        "--index", type=str, default=None,
        help="指数代码（hs300/zz500/sz50），使用指数成分股评估",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="使用指数时取前N只成分股，默认10",
    )
    parser.add_argument(
        "--method", type=str, default="all",
        choices=["kmv", "copula", "var", "scoring", "all"],
        help="分析方法: kmv/copula/var/scoring/all（默认all）",
    )
    parser.add_argument(
        "--output", type=str, default="output",
        help="输出目录，默认 output/",
    )
    parser.add_argument(
        "--rf", type=float, default=None,
        help="无风险利率（小数，如0.025表示2.5%%），默认自动获取",
    )
    parser.add_argument(
        "--sim", type=int, default=100000,
        help="蒙特卡洛模拟次数，默认100000",
    )
    return parser.parse_args()


def main():
    """主函数。"""
    args = parse_args()

    _print(Panel.fit(
        "[bold blue]信用风险建模工具 v1.0.0[/bold blue]\n"
        "KMV-Merton | Copula | Credit VaR | Altman Z-Score",
        border_style="blue",
    ) if RICH_AVAILABLE else "=" * 60 + "\n信用风险建模工具 v1.0.0\n" + "=" * 60)

    # 获取股票列表
    tickers = []
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    elif args.index:
        _print(f"\n[blue]获取指数 {args.index} 成分股...[/blue]" if RICH_AVAILABLE
               else f"\n获取指数 {args.index} 成分股...")
        try:
            tickers = data.get_index_constituents(args.index, top_n=args.top)
            _print(f"  获取到 {len(tickers)} 只成分股" if RICH_AVAILABLE
                   else f"  获取到 {len(tickers)} 只成分股")
        except Exception as e:
            _print(f"[red]获取指数成分股失败: {e}[/red]" if RICH_AVAILABLE
                   else f"获取指数成分股失败: {e}")
            sys.exit(1)
    else:
        _print("[red]请指定 --tickers 或 --index[/red]" if RICH_AVAILABLE
               else "请指定 --tickers 或 --index")
        sys.exit(1)

    if len(tickers) < 1:
        _print("[red]未获取到任何股票代码[/red]" if RICH_AVAILABLE
               else "未获取到任何股票代码")
        sys.exit(1)

    _print(f"\n[green]评估股票列表: {', '.join(tickers)}[/green]" if RICH_AVAILABLE
           else f"\n评估股票列表: {', '.join(tickers)}")

    method = args.method
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # ==================== 第1步: 获取财务数据 ====================
    _print("\n[bold]步骤 1/5: 获取财务数据[/bold]" if RICH_AVAILABLE
           else "\n步骤 1/5: 获取财务数据")

    panels = {}
    if RICH_AVAILABLE:
        with _progress_context() as progress:
            task = progress.add_task("获取数据中...", total=len(tickers))
            for ticker in tickers:
                try:
                    panel = data.build_credit_panel(ticker)
                    panels[panel["ticker"]] = panel
                    progress.update(task, description=f"已获取 {panel['name']}({ticker})")
                except Exception as e:
                    _print(f"  [red]获取 {ticker} 失败: {e}[/red]")
                progress.advance(task)
    else:
        for ticker in tickers:
            try:
                panel = data.build_credit_panel(ticker)
                panels[panel["ticker"]] = panel
                print(f"  已获取 {panel['name']}({ticker})")
            except Exception as e:
                print(f"  获取 {ticker} 失败: {e}")

    if len(panels) < 1:
        _print("[red]未能获取任何公司的有效数据[/red]" if RICH_AVAILABLE
               else "未能获取任何公司的有效数据")
        sys.exit(1)

    _print(f"\n[green]成功获取 {len(panels)} 家公司数据[/green]" if RICH_AVAILABLE
           else f"\n成功获取 {len(panels)} 家公司数据")

    # ==================== 第2步: KMV模型分析 ====================
    kmv_results = None
    if method in ("kmv", "all"):
        _print("\n[bold]步骤 2/5: KMV-Merton模型分析[/bold]" if RICH_AVAILABLE
               else "\n步骤 2/5: KMV-Merton模型分析")

        rf = args.rf
        if rf is None:
            rf = kmv_model.get_risk_free_rate()
            _print(f"  无风险利率: {rf*100:.2f}%" if RICH_AVAILABLE
                   else f"  无风险利率: {rf*100:.2f}%")
        else:
            _print(f"  无风险利率(用户指定): {rf*100:.2f}%" if RICH_AVAILABLE
                   else f"  无风险利率(用户指定): {rf*100:.2f}%")

        kmv_results = kmv_model.kmv_assess_multi(panels, risk_free_rate=rf, T=1.0)

        if RICH_AVAILABLE:
            table = RichTable(title="KMV模型结果")
            table.add_column("代码", style="cyan")
            table.add_column("名称")
            table.add_column("资产价值(亿)", justify="right")
            table.add_column("违约点(亿)", justify="right")
            table.add_column("DD", justify="right")
            table.add_column("EDF(%)", justify="right")
            table.add_column("收敛", justify="center")

            for _, row in kmv_results.iterrows():
                table.add_row(
                    row["ticker"],
                    row["name"],
                    f"{row['asset_value']/1e8:.2f}",
                    f"{row['default_point']/1e8:.2f}",
                    f"{row['dd']:.3f}",
                    f"{row['edf']*100:.4f}",
                    "是" if row["converged"] else "否",
                )
            console.print(table)
        else:
            print(kmv_results[["ticker", "name", "dd", "edf",
                                "asset_value", "default_point"]].to_string())

        # 保存KMV结果
        kmv_results.to_csv(os.path.join(output_dir, "kmv_results.csv"),
                            index=False, encoding="utf-8-sig")

    # ==================== 第3步: 信用评分 ====================
    scoring_results = None
    if method in ("scoring", "all"):
        _print("\n[bold]步骤 3/5: 信用评分分析[/bold]" if RICH_AVAILABLE
               else "\n步骤 3/5: 信用评分分析")

        scoring_results = scoring_mod.scoring_assess_multi(panels)

        if RICH_AVAILABLE:
            table = RichTable(title="信用评分结果")
            table.add_column("代码", style="cyan")
            table.add_column("名称")
            table.add_column("Z-Score", justify="right")
            table.add_column("区域")
            table.add_column("评级", style="yellow")
            table.add_column("综合评分", justify="right")
            table.add_column("评级", style="magenta")

            for _, row in scoring_results.iterrows():
                table.add_row(
                    row["ticker"],
                    row["name"],
                    f"{row['altman_z']:.3f}",
                    row["altman_zone"],
                    row["altman_rating"],
                    f"{row['comprehensive_score']:.1f}",
                    row["rating"],
                )
            console.print(table)
        else:
            print(scoring_results[["ticker", "name", "altman_z",
                                    "altman_zone", "rating"]].to_string())

        scoring_results.to_csv(os.path.join(output_dir, "scoring_results.csv"),
                                 index=False, encoding="utf-8-sig")

    # ==================== 第4步: Copula依赖结构 ====================
    copula_result = None
    if method in ("copula", "var", "all"):
        _print("\n[bold]步骤 4/5: Copula依赖结构建模[/bold]" if RICH_AVAILABLE
               else "\n步骤 4/5: Copula依赖结构建模")

        if len(panels) >= 2:
            try:
                copula_result = copula_mod.portfolio_default_correlation(
                    panels, kmv_results, copula_type="gaussian"
                )
                _print(f"  Copula类型: Gaussian" if RICH_AVAILABLE
                       else f"  Copula类型: Gaussian")
                _print(f"  联合违约概率: {copula_result['joint_pd']:.6f} "
                       f"({copula_result['joint_pd']*100:.4f}%)" if RICH_AVAILABLE
                       else f"  联合违约概率: {copula_result['joint_pd']:.6f}")

                # 保存相关矩阵
                corr_df = copula_mod.copula_correlation_heatmap_data(
                    copula_result["corr_matrix"],
                    copula_result["tickers"]
                )
                corr_df.to_csv(os.path.join(output_dir, "copula_corr_matrix.csv"),
                               encoding="utf-8-sig")

                if RICH_AVAILABLE:
                    table = RichTable(title="Copula相关矩阵")
                    headers = [""] + copula_result["tickers"]
                    for h in headers:
                        table.add_column(h, justify="center")
                    for i, t in enumerate(copula_result["tickers"]):
                        row_data = [t] + [f"{copula_result['corr_matrix'][i,j]:.3f}"
                                          for j in range(len(copula_result["tickers"]))]
                        table.add_row(*row_data)
                    console.print(table)

            except Exception as e:
                _print(f"  [yellow]Copula分析失败: {e}[/yellow]" if RICH_AVAILABLE
                       else f"  Copula分析失败: {e}")
        else:
            _print("  [yellow]需要至少2家公司才能进行Copula分析[/yellow]" if RICH_AVAILABLE
                   else "  需要至少2家公司才能进行Copula分析")

    # ==================== 第5步: 信用VaR与经济资本 ====================
    var_result = None
    if method in ("var", "all"):
        _print("\n[bold]步骤 5/5: 信用VaR与经济资本计算[/bold]" if RICH_AVAILABLE
               else "\n步骤 5/5: 信用VaR与经济资本计算")

        if copula_result is not None and kmv_results is not None:
            try:
                var_result = var_mod.portfolio_credit_assessment(
                    kmv_results=kmv_results,
                    copula_result=copula_result,
                    confidence=0.999,
                    n_simulations=args.sim,
                    random_state=42,
                )

                ps = var_result["portfolio_summary"]
                if RICH_AVAILABLE:
                    table = RichTable(title="组合信用风险摘要")
                    table.add_column("指标", style="cyan")
                    table.add_column("值", justify="right")
                    table.add_row("资产数量", str(ps["n_assets"]))
                    table.add_row("总风险暴露(亿)", f"{ps['total_ead']/1e8:.2f}")
                    table.add_row("预期损失EL(亿)", f"{ps['total_el']/1e8:.4f}")
                    table.add_row("意外损失UL(亿)", f"{ps['total_ul']/1e8:.4f}")
                    table.add_row("信用VaR(亿)", f"{ps['credit_var']/1e8:.4f}")
                    table.add_row("经济资本EC(亿)", f"{ps['ec']/1e8:.4f}")
                    table.add_row("蒙特卡洛EL(亿)", f"{ps['mc_el']/1e8:.4f}")
                    table.add_row("蒙特卡洛VaR(亿)", f"{ps['mc_credit_var']/1e8:.4f}")
                    table.add_row("最坏损失(亿)", f"{ps['mc_worst_loss']/1e8:.4f}")
                    console.print(table)

                    # 风险贡献表
                    contrib = var_result["contributions"]
                    table2 = RichTable(title="边际风险贡献")
                    table2.add_column("代码", style="cyan")
                    table2.add_column("名称")
                    table2.add_column("EL(万)", justify="right")
                    table2.add_column("UL(万)", justify="right")
                    table2.add_column("MRC(万)", justify="right")
                    for _, row in contrib.iterrows():
                        table2.add_row(
                            str(row.get("ticker", "")),
                            str(row.get("name", "")),
                            f"{row['EL']/1e4:.2f}",
                            f"{row['UL']/1e4:.2f}",
                            f"{row['MRC']/1e4:.2f}",
                        )
                    console.print(table2)
                else:
                    print(f"  总EAD: {ps['total_ead']/1e8:.2f}亿")
                    print(f"  EL: {ps['total_el']/1e8:.4f}亿")
                    print(f"  UL: {ps['total_ul']/1e8:.4f}亿")
                    print(f"  信用VaR: {ps['credit_var']/1e8:.4f}亿")
                    print(f"  经济资本: {ps['ec']/1e8:.4f}亿")

                # 保存风险贡献
                var_result["contributions"].to_csv(
                    os.path.join(output_dir, "risk_contributions.csv"),
                    index=False, encoding="utf-8-sig"
                )

            except Exception as e:
                _print(f"  [yellow]VaR计算失败: {e}[/yellow]" if RICH_AVAILABLE
                       else f"  VaR计算失败: {e}")
        else:
            _print("  [yellow]需要Copula分析和KMV结果才能计算VaR[/yellow]" if RICH_AVAILABLE
                   else "  需要Copula分析和KMV结果才能计算VaR")

    # ==================== 生成HTML报告 ====================
    if method == "all" and kmv_results is not None and scoring_results is not None:
        _print("\n[bold]生成HTML报告...[/bold]" if RICH_AVAILABLE
               else "\n生成HTML报告...")

        # 行业统计
        industry_stats = kmv_model.dd_industry_comparison(kmv_results)
        if industry_stats is None or len(industry_stats) == 0:
            industry_stats = scoring_mod.industry_benchmark_comparison(scoring_results)

        try:
            html_path = report_mod.generate_html_report(
                kmv_results=kmv_results,
                scoring_results=scoring_results,
                copula_result=copula_result,
                var_result=var_result,
                industry_stats=industry_stats,
                output_dir=output_dir,
            )
            _print(f"\n[green bold]报告已生成: {html_path}[/green bold]" if RICH_AVAILABLE
                   else f"\n报告已生成: {html_path}")
        except Exception as e:
            _print(f"\n[red]报告生成失败: {e}[/red]" if RICH_AVAILABLE
                   else f"\n报告生成失败: {e}")
    else:
        _print("\n[yellow]仅 --method all 时生成完整HTML报告[/yellow]" if RICH_AVAILABLE
               else "\n仅 --method all 时生成完整HTML报告")

    _print("\n[bold green]评估完成[/bold green]" if RICH_AVAILABLE
           else "\n评估完成")


if __name__ == "__main__":
    main()
