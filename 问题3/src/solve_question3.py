#!/usr/bin/env python3
"""Solve Question 3 of B problem and generate paper-ready artifacts."""

from __future__ import annotations

import csv
import importlib.util
import itertools
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
Q3_DIR = ROOT / "问题3"
SRC_DIR = Q3_DIR / "src"
IMG_DIR = Q3_DIR / "img"
DOC_DIR = Q3_DIR / "docs"
TABLE_DIR = DOC_DIR / "tables"
Q2_SRC = ROOT / "问题2" / "src" / "solve_question2.py"
Q2_TABLE_DIR = ROOT / "问题2" / "docs" / "tables"

MPL_CACHE_BOOTSTRAP = Q3_DIR / ".matplotlib-cache"
MPL_CACHE_BOOTSTRAP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_BOOTSTRAP))
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def import_q2_module():
    spec = importlib.util.spec_from_file_location("q2_solver", Q2_SRC)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {Q2_SRC}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["q2_solver"] = module
    spec.loader.exec_module(module)
    return module


q2 = import_q2_module()

COMMUNITIES = q2.COMMUNITIES
TYPE_LABELS = q2.TYPE_LABELS
SERVICE_ORDER = q2.SERVICE_ORDER
PAID_SERVICES = [s for s in SERVICE_ORDER if s != "紧急救助"]
DAYS_PER_MONTH = 30.0
PRICE_GRID = [round(0.30 + 0.001 * i, 3) for i in range(701)]
DAILY_SUBSIDY_CAP = {"小型": 1000.0, "中型": 1800.0, "大型": 2600.0}


@dataclass
class StationPricingResult:
    station: str
    scale: str
    covered: list[str]
    prices: dict[str, float]
    price_multipliers: dict[str, float]
    service_price_score: dict[str, float]
    community_satisfaction: dict[str, float]
    community_price_score: dict[str, float]
    community_response_score: dict[str, float]
    community_effective_monthly: dict[str, float]
    type_access: dict[str, float]
    type_economic_access: dict[str, float]
    monthly_demand: dict[str, dict[str, dict[str, int]]]
    effective_by_service: dict[str, float]
    effective_daily: float
    theta: float
    response_score: float
    subsidy: float
    subsidy_before_cap: float
    service_profit: float
    fixed_cost: float
    net_profit: float
    profit_rate: float
    weighted_satisfaction: float
    weighted_price_score: float
    weighted_price_multiplier: float


def choose_font() -> None:
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC", "Arial Unicode MS"]
    for font in preferred:
        try:
            fm.findfont(font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font]
            break
        except ValueError:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def export_csv(path: Path, rows: list[dict[str, Any]], headers: list[str] | None = None) -> None:
    if not rows and headers is None:
        return
    if headers is None:
        headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def price_satisfaction(price: float, base_price: float) -> float:
    if base_price <= 0:
        return 1.0
    ratio = price / base_price
    if ratio <= 1.0:
        return 1.0
    if ratio <= 1.1:
        return 0.9
    if ratio <= 1.2:
        return 0.75
    return 0.6


def read_fixed_q2_plan() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    station_rows = read_csv_dicts(Q2_TABLE_DIR / "q2_station_plan.csv")
    assignment_rows = read_csv_dicts(Q2_TABLE_DIR / "q2_community_assignment.csv")
    stations = {
        row["站点小区"]: {
            "scale": row["规模"],
            "covered": [x for x in row["覆盖小区"].split("、") if x],
            "capacity": float(row["日容量_人次"]),
            "distance_score": {},
        }
        for row in station_rows
    }
    assignments = {}
    for row in assignment_rows:
        community = row["小区"]
        station = row["分配站点"]
        if station == "未覆盖" or not station:
            continue
        assignments[community] = {
            "station": station,
            "distance": float(row["距离_米"]),
            "distance_score": float(row["距离满意度"]),
        }
        stations[station]["distance_score"][community] = float(row["距离满意度"])
    return stations, assignments


def settle_response(
    station: str,
    scale: str,
    covered: list[str],
    capacity: float,
    distance_score: dict[str, float],
    demand_counts: dict[str, dict[str, dict[str, int]]],
    service_price_score: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float, float]:
    response = 1.0
    last_payload = None
    seen: set[float] = set()
    for _ in range(30):
        community_price_score: dict[str, float] = {}
        satisfaction: dict[str, float] = {}
        effective_monthly: dict[str, float] = {}
        for community in covered:
            total_q = 0
            score_sum = 0.0
            for service in SERVICE_ORDER:
                q_service = sum(demand_counts[community][service][t] for t in TYPE_LABELS)
                total_q += q_service
                score_sum += q_service * service_price_score[service]
            p_score = score_sum / total_q if total_q > 0 else 1.0
            s = 0.2 * distance_score[community] + 0.3 * response + 0.5 * p_score
            community_price_score[community] = p_score
            satisfaction[community] = s
            effective_monthly[community] = total_q * s
        effective_daily = sum(effective_monthly.values()) / DAYS_PER_MONTH
        theta = effective_daily / capacity if capacity else float("inf")
        new_response = q2.response_satisfaction(theta)
        last_payload = (satisfaction, community_price_score, effective_monthly, theta, new_response)
        if abs(new_response - response) < 1e-12:
            return last_payload
        if new_response in seen:
            response = min(response, new_response, *seen)
            continue
        seen.add(response)
        response = new_response
    return last_payload


def evaluate_station_prices(
    station: str,
    scale: str,
    covered: list[str],
    capacity: float,
    distance_score: dict[str, float],
    price_vector: dict[str, float],
    data: dict[str, Any],
    enforce_constraints: bool = True,
) -> StationPricingResult | None:
    population = data["population"]
    year5 = data["year5"]
    demand = data["demand"]
    base_price = data["price"]
    direct_cost = data["direct_cost"]
    station_costs = data["station_costs"]
    alpha = {"自理": 0.20, "半失能": 0.25, "失能": 0.30}

    prices = dict(price_vector)
    prices["紧急救助"] = 0.0
    service_score = {s: price_satisfaction(prices[s], base_price[s]) for s in SERVICE_ORDER}

    demand_counts = {
        community: {service: {t: 0 for t in TYPE_LABELS} for service in SERVICE_ORDER}
        for community in covered
    }
    economic_access_values = {t: [] for t in TYPE_LABELS}
    type_theoretical = {t: 0.0 for t in TYPE_LABELS}
    type_effective = {t: 0.0 for t in TYPE_LABELS}

    for community in covered:
        income = population[community]["人均月收入"]
        for elder_type in TYPE_LABELS:
            theoretical_cost = sum(prices[s] * demand[s][elder_type] for s in SERVICE_ORDER)
            limit = alpha[elder_type] * income
            scale_factor = min(1.0, limit / theoretical_cost) if theoretical_cost > 0 else 1.0
            realized_cost = 0.0
            theoretical_count = 0.0
            for service in SERVICE_ORDER:
                theoretical_count += year5[community][elder_type] * demand[service][elder_type]
                q_adj = demand[service][elder_type] * scale_factor
                q_total = q2.round_half_up(year5[community][elder_type] * q_adj)
                demand_counts[community][service][elder_type] = q_total
                realized_cost += prices[service] * q_adj
            type_theoretical[elder_type] += theoretical_count
            economic_access_values[elder_type].append(max(0.0, 1.0 - realized_cost / limit) if limit > 0 else 0.0)

    settled = settle_response(station, scale, covered, capacity, distance_score, demand_counts, service_score)
    if settled is None:
        return None
    satisfaction, community_price_score, effective_monthly, theta, response_score = settled
    effective_by_service = {s: 0.0 for s in SERVICE_ORDER}
    service_profit = 0.0
    subsidy_count = 0.0
    for community in covered:
        for service in SERVICE_ORDER:
            q_service = sum(demand_counts[community][service][t] for t in TYPE_LABELS)
            q_eff = q_service * satisfaction[community]
            effective_by_service[service] += q_eff
            service_profit += 12.0 * q_eff * (prices[service] - direct_cost[service])
            if service != "紧急救助":
                subsidy_count += q_eff
        for elder_type in TYPE_LABELS:
            q_type = sum(demand_counts[community][service][elder_type] for service in SERVICE_ORDER)
            type_effective[elder_type] += q_type * satisfaction[community]

    subsidy_before_cap = 12.0 * 2.0 * subsidy_count
    subsidy = min(365.0 * DAILY_SUBSIDY_CAP[scale], subsidy_before_cap)
    fixed_cost = 365.0 * station_costs[scale]["日固定成本"] + 10000.0 * station_costs[scale]["建设成本"] / 20.0
    net_profit = service_profit + subsidy - fixed_cost
    profit_rate = net_profit / fixed_cost if fixed_cost else 0.0
    if enforce_constraints and (profit_rate < -1e-9 or profit_rate > 0.08 + 1e-9):
        return None

    elderly_weight = sum(year5[c]["老人总数"] for c in covered)
    weighted_satisfaction = sum(year5[c]["老人总数"] * satisfaction[c] for c in covered) / elderly_weight
    weighted_price_score = sum(year5[c]["老人总数"] * community_price_score[c] for c in covered) / elderly_weight
    nonurgent_effective = sum(effective_by_service[s] for s in PAID_SERVICES)
    weighted_price_multiplier = (
        sum(effective_by_service[s] * (prices[s] / base_price[s]) for s in PAID_SERVICES) / nonurgent_effective
        if nonurgent_effective
        else 0.0
    )
    type_access = {
        t: type_effective[t] / type_theoretical[t] if type_theoretical[t] > 0 else 0.0 for t in TYPE_LABELS
    }
    type_economic_access = {
        t: sum(economic_access_values[t]) / len(economic_access_values[t]) if economic_access_values[t] else 0.0
        for t in TYPE_LABELS
    }
    multipliers = {s: prices[s] / base_price[s] if base_price[s] else 0.0 for s in SERVICE_ORDER}

    return StationPricingResult(
        station=station,
        scale=scale,
        covered=covered,
        prices=prices,
        price_multipliers=multipliers,
        service_price_score=service_score,
        community_satisfaction=satisfaction,
        community_price_score=community_price_score,
        community_response_score={c: response_score for c in covered},
        community_effective_monthly=effective_monthly,
        type_access=type_access,
        type_economic_access=type_economic_access,
        monthly_demand=demand_counts,
        effective_by_service=effective_by_service,
        effective_daily=sum(effective_monthly.values()) / DAYS_PER_MONTH,
        theta=theta,
        response_score=response_score,
        subsidy=subsidy,
        subsidy_before_cap=subsidy_before_cap,
        service_profit=service_profit,
        fixed_cost=fixed_cost,
        net_profit=net_profit,
        profit_rate=profit_rate,
        weighted_satisfaction=weighted_satisfaction,
        weighted_price_score=weighted_price_score,
        weighted_price_multiplier=weighted_price_multiplier,
    )


def optimize_station(station: str, info: dict[str, Any], data: dict[str, Any]) -> StationPricingResult:
    base_price = data["price"]
    best: StationPricingResult | None = None
    checked = 0
    feasible = 0
    for multiplier in PRICE_GRID:
        checked += 1
        price_vector = {service: base_price[service] * multiplier for service in PAID_SERVICES}
        result = evaluate_station_prices(
            station=station,
            scale=info["scale"],
            covered=info["covered"],
            capacity=info["capacity"],
            distance_score=info["distance_score"],
            price_vector=price_vector,
            data=data,
        )
        if result is None:
            continue
        feasible += 1
        key = (
            result.weighted_satisfaction,
            result.weighted_price_score,
            -abs(result.profit_rate - 0.04),
            -result.weighted_price_multiplier,
            -result.net_profit,
        )
        if best is None:
            best = result
            best_key = key
        elif key > best_key:
            best = result
            best_key = key
    if best is None:
        raise RuntimeError(f"No feasible pricing found for station {station}")
    print(f"{station}: checked={checked}, feasible={feasible}, best_PR={best.profit_rate:.4%}")
    return best


def solve_question3() -> dict[str, Any]:
    population, transition, demand, price, direct_cost, station_costs, distance = q2.read_inputs()
    year5 = q2.forecast_population(population, transition)
    stations, assignments = read_fixed_q2_plan()
    data = {
        "population": population,
        "transition": transition,
        "demand": demand,
        "price": price,
        "direct_cost": direct_cost,
        "station_costs": station_costs,
        "distance": distance,
        "year5": year5,
    }
    results = {station: optimize_station(station, info, data) for station, info in stations.items()}
    return {"data": data, "stations": stations, "assignments": assignments, "results": results}


def station_price_rows(solution) -> list[dict[str, Any]]:
    rows = []
    for station, result in solution["results"].items():
        row = {"站点": station, "规模": result.scale}
        for service in SERVICE_ORDER:
            row[f"{service}价格"] = result.prices[service]
        for service in SERVICE_ORDER:
            row[f"{service}价格满意度"] = result.service_price_score[service]
        row["加权价格倍率"] = result.weighted_price_multiplier
        row["价格满意度"] = result.weighted_price_score
        rows.append(row)
    return rows


def station_financial_rows(solution) -> list[dict[str, Any]]:
    rows = []
    for station, result in solution["results"].items():
        rows.append(
            {
                "站点": station,
                "规模": result.scale,
                "服务总利润_元": result.service_profit,
                "政府补贴_元": result.subsidy,
                "补贴封顶前_元": result.subsidy_before_cap,
                "年运营成本_元": result.fixed_cost,
                "年度净利润_元": result.net_profit,
                "利润率": result.profit_rate,
                "日有效服务人次": result.effective_daily,
                "利用率": result.theta,
                "响应满意度": result.response_score,
            }
        )
    rows.append(
        {
            "站点": "合计",
            "规模": "",
            "服务总利润_元": sum(r.service_profit for r in solution["results"].values()),
            "政府补贴_元": sum(r.subsidy for r in solution["results"].values()),
            "补贴封顶前_元": sum(r.subsidy_before_cap for r in solution["results"].values()),
            "年运营成本_元": sum(r.fixed_cost for r in solution["results"].values()),
            "年度净利润_元": sum(r.net_profit for r in solution["results"].values()),
            "利润率": "",
            "日有效服务人次": sum(r.effective_daily for r in solution["results"].values()),
            "利用率": "",
            "响应满意度": "",
        }
    )
    return rows


def community_rows(solution) -> list[dict[str, Any]]:
    data = solution["data"]
    year5 = data["year5"]
    rows = []
    for community in COMMUNITIES:
        row = {
            "小区": community,
            "第5年老人总数": year5[community]["老人总数"],
            "服务站": "未覆盖",
            "综合满意度": 0.0,
            "价格满意度": 0.0,
            "响应满意度": 0.0,
            "月有效服务人次": 0.0,
        }
        for station, result in solution["results"].items():
            if community in result.covered:
                row.update(
                    {
                        "服务站": station,
                        "综合满意度": result.community_satisfaction[community],
                        "价格满意度": result.community_price_score[community],
                        "响应满意度": result.community_response_score[community],
                        "月有效服务人次": result.community_effective_monthly[community],
                    }
                )
                break
        rows.append(row)
    return rows


def access_rows(solution) -> list[dict[str, Any]]:
    data = solution["data"]
    year5 = data["year5"]
    rows = []
    for elder_type in TYPE_LABELS:
        covered_population = 0
        service_num = 0.0
        econ_num = 0.0
        geo_num = 0.0
        for station, result in solution["results"].items():
            for community in result.covered:
                weight = year5[community][elder_type]
                covered_population += weight
                service_num += weight * result.type_access[elder_type]
                econ_num += weight * result.type_economic_access[elder_type]
                # Distance score is a geography proxy for the assigned community.
                geo_num += weight * solution["stations"][station]["distance_score"][community]
        rows.append(
            {
                "老人类型": elder_type,
                "覆盖人数": covered_population,
                "服务可及性": service_num / covered_population if covered_population else 0.0,
                "经济可及性": econ_num / covered_population if covered_population else 0.0,
                "地理可及性": geo_num / covered_population if covered_population else 0.0,
            }
        )
    return rows


def summary_rows(solution) -> list[dict[str, Any]]:
    data = solution["data"]
    year5 = data["year5"]
    covered = set()
    for result in solution["results"].values():
        covered.update(result.covered)
    covered_elderly = sum(year5[c]["老人总数"] for c in covered)
    total_elderly = sum(year5[c]["老人总数"] for c in COMMUNITIES)
    community_result_rows = community_rows(solution)
    avg_satisfaction = (
        sum(year5[row["小区"]]["老人总数"] * row["综合满意度"] for row in community_result_rows)
        / covered_elderly
    )
    avg_price_score = (
        sum(year5[row["小区"]]["老人总数"] * row["价格满意度"] for row in community_result_rows)
        / covered_elderly
    )
    return [
        {"指标": "固定站点数量", "数值": len(solution["results"])},
        {"指标": "覆盖老人数量", "数值": covered_elderly},
        {"指标": "服务覆盖率", "数值": covered_elderly / total_elderly},
        {"指标": "覆盖人口加权满意度", "数值": avg_satisfaction},
        {"指标": "覆盖人口加权价格满意度", "数值": avg_price_score},
        {"指标": "政府补贴总额（元）", "数值": sum(r.subsidy for r in solution["results"].values())},
        {"指标": "年度净利润合计（元）", "数值": sum(r.net_profit for r in solution["results"].values())},
    ]


def format_number(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], headers: list[str] | None = None, digits: int = 4) -> str:
    if not rows:
        return ""
    if headers is None:
        headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_number(row.get(h, ""), digits) for h in headers) + " |")
    return "\n".join(lines)


def plot_outputs(solution) -> None:
    choose_font()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    prices = station_price_rows(solution)
    financial = station_financial_rows(solution)[:-1]
    communities = community_rows(solution)
    access = access_rows(solution)

    x = np.arange(len(solution["results"]))
    services = PAID_SERVICES
    width = 0.14
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for idx, service in enumerate(services):
        ax.bar(
            x + (idx - 2) * width,
            [row[f"{service}价格"] for row in prices],
            width=width,
            label=service,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([row["站点"] for row in prices])
    ax.set_ylabel("价格（元/次）")
    ax.set_title("各服务站最优服务定价", fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.9)
    ax.legend(ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "q3_station_prices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8.8, 5.0))
    ax2 = ax1.twinx()
    stations = [row["站点"] for row in financial]
    ax1.bar(np.arange(len(stations)) - 0.18, [row["政府补贴_元"] / 10000 for row in financial], 0.36, color="#4C78A8", label="政府补贴")
    ax1.bar(np.arange(len(stations)) + 0.18, [row["年度净利润_元"] / 10000 for row in financial], 0.36, color="#F58518", label="年度净利润")
    ax2.plot(np.arange(len(stations)), [row["利润率"] * 100 for row in financial], color="#E45756", marker="o", linewidth=2.2, label="利润率")
    ax2.axhline(8, color="#9CA3AF", linestyle="--", linewidth=1.5)
    ax1.set_xticks(np.arange(len(stations)))
    ax1.set_xticklabels(stations)
    ax1.set_ylabel("金额（万元）")
    ax2.set_ylabel("利润率（%）")
    ax1.set_title("补贴、净利润与利润率", fontsize=15, fontweight="bold", pad=12)
    ax1.grid(axis="y", color="#E5E7EB", linewidth=0.9)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "q3_profit_subsidy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar([row["小区"] for row in communities], [row["综合满意度"] for row in communities], color="#4C78A8", width=0.68, label="综合满意度")
    ax.plot([row["小区"] for row in communities], [row["价格满意度"] for row in communities], color="#E45756", marker="o", linewidth=2.0, label="价格满意度")
    ax.set_ylim(0.55, 1.03)
    ax.set_ylabel("得分")
    ax.set_title("各小区综合满意度与价格满意度", fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.9)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "q3_community_satisfaction_price.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    labels = [row["老人类型"] for row in access]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(x - 0.24, [row["服务可及性"] for row in access], 0.24, label="服务可及性", color="#4C78A8")
    ax.bar(x, [row["经济可及性"] for row in access], 0.24, label="经济可及性", color="#54A24B")
    ax.bar(x + 0.24, [row["地理可及性"] for row in access], 0.24, label="地理可及性", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("可及性指标")
    ax.set_title("不同类型老人服务可及性", fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.9)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "q3_accessibility_by_type.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_paper(solution) -> str:
    summary = summary_rows(solution)
    prices = station_price_rows(solution)
    financial = station_financial_rows(solution)
    communities = community_rows(solution)
    access = access_rows(solution)

    price_headers = ["站点", "规模"] + [f"{s}价格" for s in SERVICE_ORDER] + ["加权价格倍率", "价格满意度"]
    financial_headers = ["站点", "规模", "服务总利润_元", "政府补贴_元", "年运营成本_元", "年度净利润_元", "利润率", "日有效服务人次", "响应满意度"]
    community_headers = ["小区", "第5年老人总数", "服务站", "综合满意度", "价格满意度", "响应满意度", "月有效服务人次"]

    paper = rf"""# 第三问：服务定价与政府补贴优化

## 1 问题重述与建模思路

第三问固定第二问得到的最优站点方案，即 \(D\) 小区中型站、\(G\) 小区中型站、\(J\) 小区大型站，并保持第二问的覆盖关系不变。在此基础上，允许各服务站对助餐、日间照料、上门护理、康复理疗、助浴五类非紧急服务自主定价，紧急救助保持公益免费。政府对非紧急服务按实际有效服务人次补贴 2 元/人次，并设置单站每日补贴上限。优化目标是在满足机构“保本微利、利润率不超过 8%”的前提下，最大化老人满意度。

本文沿用 `docs/问题分析.md` 中建议的“候选价格档位枚举 + 约束过滤”方法。由于站点和覆盖关系已经固定，各服务站之间不存在容量共享或价格联动约束，因此可对每个站点独立枚举价格向量，再汇总得到全局方案。考虑到第二问的站点已接近满负荷，第三问不再将价格变化后的容量作为硬筛选条件，而是通过利用率对应的响应满意度 \(S_2\) 反映高负荷对服务体验的影响。

## 2 模型建立

### 2.1 决策变量

给定问题二最优站点集合 \(J^\star=\{{D,G,J\}}\)，对每个站点 \(j\in J^\star\) 和每个服务 \(s\in S\)，设服务价格为
\[
p_{{j,s}}\ge 0.
\]
紧急救助为公益免费，因此
\[
p_{{j,e}}=0,\quad j\in J^\star.
\]

### 2.2 价格满意度与需求调整

附件5给出的价格满意度规则为
\[
S_{{3,j,s}}=
\begin{{cases}}
1.00,&p_{{j,s}}\le p_s^0,\\
0.90,&p_s^0<p_{{j,s}}\le1.1p_s^0,\\
0.75,&1.1p_s^0<p_{{j,s}}\le1.2p_s^0,\\
0.60,&p_{{j,s}}>1.2p_s^0.
\end{{cases}}
\]
在站点 \(j\) 的价格下，小区 \(i\)、老人类型 \(t\) 的理论月费用为
\[
E_{{i,t,j}}(p)=\sum_{{s\in S}}p_{{j,s}}q_{{s,t}}^0.
\]
消费上限为
\[
L_{{i,t}}=\alpha_tM_i.
\]
价格对应的需求削减系数为
\[
\lambda_{{i,t,j}}(p)=\min\left\{{1,\frac{{L_{{i,t}}}}{{E_{{i,t,j}}(p)}}\right\}}.
\]
于是价格约束后的月需求为
\[
Q_{{i,s,t,5}}(p)=N_{{i,5}}^t\operatorname{{round}}\left(\lambda_{{i,t,j}}(p)q_{{s,t}}^0\right),
\quad u_{{i,j}}=1.
\]

小区 \(i\) 的需求加权价格满意度为
\[
\overline S_{{3,i,j}}=
\frac{{\sum_{{s,t}}Q_{{i,s,t,5}}(p)S_{{3,j,s}}}}
{{\sum_{{s,t}}Q_{{i,s,t,5}}(p)}}.
\]
综合满意度为
\[
S_{{ij}}(p)=0.2S_{{1,ij}}+0.3S_{{2,j}}(p)+0.5\overline S_{{3,i,j}}.
\]

### 2.3 补贴与利润率约束

实际有效服务人次为
\[
Q_{{i,s,t,5}}^{{\mathrm{{eff}}}}(p)=Q_{{i,s,t,5}}(p)S_{{ij}}(p).
\]
非紧急服务补贴为 2 元/人次，并受每日上限约束：
\[
H_j(p)=\min\left\{{365b_{{k(j)}},\,
12\cdot2\sum_{{i,t}}\sum_{{s\ne e}}u_{{i,j}}Q_{{i,s,t,5}}^{{\mathrm{{eff}}}}(p)
\right\}}.
\]
其中中型站 \(b_k=1800\)，大型站 \(b_k=2600\)。

服务总利润为
\[
G_j(p)=12\sum_{{i,t,s}}u_{{i,j}}Q_{{i,s,t,5}}^{{\mathrm{{eff}}}}(p)(p_{{j,s}}-c_s).
\]
年运营成本总额为
\[
A_j=365F_{{k(j)}}+\frac{{10000B_{{k(j)}}}}{{20}}.
\]
利润率定义为
\[
\mathrm{{PR}}_j(p)=\frac{{G_j(p)+H_j(p)-A_j}}{{A_j}}.
\]
保本微利约束写为
\[
0\le\mathrm{{PR}}_j(p)\le 8\%,\quad j\in J^\star.
\]

### 2.4 优化目标

最大化覆盖老人加权综合满意度：
\[
\max_p\ \overline S(p)=
\frac{{\sum_iN_{{i,5}}\sum_j u_{{i,j}}S_{{ij}}(p)}}
{{\sum_iN_{{i,5}}\sum_j u_{{i,j}}}}.
\]
当多组价格满意度相同且均满足利润率约束时，本文优先选择加权价格倍率更低、利润率更接近 \(4\%\) 的方案，使老人负担更低且机构仍保留微利空间。

## 3 求解算法

为降低维度并保持站点内部价格体系简洁，本文采用 `docs/问题分析.md` 中的降维策略：每个服务站使用一个整体价格倍率 \(r_j\)，即同一站点内五类非紧急服务按相同倍率调整。候选价格倍率取
\[
r_j\in\{{0.300,0.301,\ldots,0.999,1.000\}},
\quad p_{{j,s}}=r_jp_s^0.
\]
对每个站点独立枚举 701 个价格倍率。每个价格倍率按如下步骤评价：

1. 根据价格计算各小区、各老人类型的消费约束需求；
2. 根据价格满意度、距离满意度和响应满意度迭代求得 \(S_{{ij}}\)；
3. 计算实际有效服务人次、政府补贴、服务总利润和利润率；
4. 剔除利润率不在 \([0,8\%]\) 内的方案；
5. 在可行价格向量中选择满意度最高的方案。

## 4 求解结果

核心指标如下。

{markdown_table(summary, ["指标", "数值"])}

### 4.1 最优定价

{markdown_table(prices, price_headers, digits=2)}

![各服务站最优服务定价](../img/q3_station_prices.png)

### 4.2 利润、利润率与补贴

{markdown_table(financial, financial_headers, digits=4)}

![补贴、净利润与利润率](../img/q3_profit_subsidy.png)

### 4.3 小区满意度与价格满意度

{markdown_table(communities, community_headers, digits=4)}

![各小区综合满意度与价格满意度](../img/q3_community_satisfaction_price.png)

## 5 不同类型老人服务可及性分析

本文从三方面刻画可及性：经济可及性表示消费上限的剩余空间，地理可及性用距离满意度表示，服务可及性用实际有效服务量与理论需求量之比表示。结果如下。

{markdown_table(access, ["老人类型", "覆盖人数", "服务可及性", "经济可及性", "地理可及性"], digits=4)}

![不同类型老人服务可及性](../img/q3_accessibility_by_type.png)

从结果看，自理老人服务频次低、价格敏感性较弱，经济可及性最高；半失能老人和失能老人护理、康复、助浴等需求较高，消费约束更容易发挥作用。补贴和降价以后，三类老人的价格满意度均保持在 1.00，但失能老人仍因需求基数高、服务消耗大而表现出更低的服务可及性。因此后续政策若进一步倾斜，应优先面向失能和半失能老人，提高护理、康复、助浴服务的专项补贴强度。

## 6 结论

在固定第二问站点和覆盖关系的条件下，本文通过枚举价格档位得到满足利润率约束的最优定价方案。各站点利润率均处于 \(0\%\) 到 \(8\%\) 之间，政府补贴有效降低了服务价格，同时使价格满意度保持满分。最终覆盖人口加权综合满意度为 {summary[3]["数值"]:.4f}，覆盖人口加权价格满意度为 {summary[4]["数值"]:.4f}，政府年度补贴总额为 {summary[5]["数值"]:.2f} 元。该结果为后续第四问的参数敏感性分析提供了基准定价和补贴方案。
"""
    return paper


def export_outputs(solution) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    export_csv(TABLE_DIR / "q3_summary.csv", summary_rows(solution))
    export_csv(TABLE_DIR / "q3_station_prices.csv", station_price_rows(solution))
    export_csv(TABLE_DIR / "q3_station_financials.csv", station_financial_rows(solution))
    export_csv(TABLE_DIR / "q3_community_satisfaction.csv", community_rows(solution))
    export_csv(TABLE_DIR / "q3_accessibility_by_type.csv", access_rows(solution))

    plot_outputs(solution)
    (DOC_DIR / "第三问论文.md").write_text(build_paper(solution), encoding="utf-8")


def main() -> None:
    solution = solve_question3()
    export_outputs(solution)
    print(f"Generated paper: {DOC_DIR / '第三问论文.md'}")
    print(f"Generated tables: {TABLE_DIR}")
    print(f"Generated images: {IMG_DIR}")
    for station, result in solution["results"].items():
        print(
            f"{station}: satisfaction={result.weighted_satisfaction:.4f}, "
            f"price_score={result.weighted_price_score:.4f}, "
            f"profit_rate={result.profit_rate:.4%}, subsidy={result.subsidy:.2f}"
        )


if __name__ == "__main__":
    main()
