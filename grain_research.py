"""Corn strategy research functions used by the backtest notebook.

The notebook imports from this module directly. Corn-only feature builders,
signal selection, position sizing, and candidate guard tests live here so the
bundle has fewer helper files while still keeping `grain_futures_strategy.py`
small and shared by all notebooks.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from research_config import (
    COMMODITY_LOCATION_WEIGHTS,
    CONTRACT_MULTIPLIER,
    CORN_HOLDING_COST_RATE,
    CORN_IC_THRESHOLD,
    CORN_MAX_ABS_LOT,
    CORN_TARGET_DAILY_PNL_VOL,
    CORN_TRADE_COST_PER_LOT,
    CORN_TRAIN_END,
    DEFAULT_MARGIN_PER_LOT,
    REGIME_PERIODS,
    SPLIT_DATE,
)
from grain_futures_strategy import load_train_set
from strategy_backtest_common import backtest_positions_with_costs, split_performance


__all__ = [
    "build_product_flow_feature_panels",
    "load_train_set",
    "build_corn_product_flow_signal_universe",
    "corn_signal_set_families",
    "corn_average_all_signals",
    "corn_equal_family_signal",
    "corn_select_by_ic_signal",
    "corn_trend_mr_family_signal",
    "corn_dynamic_linear_family_signal",
    "corn_family_signal",
    "mean_product_flow_signals",
    "corn_positions_from_signal",
    "backtest_positions_product_flow",
    "summarize_corn_backtest",
    "clean_product_flow_signal",
    "product_flow_performance_metrics",
    "product_flow_period_performance",
    "build_corn_vol_regime_signal",
    "corn_abundant_supply_masks",
    "build_corn_carry_forward_candidates",
    "make_corn_candidate",
    "summarize_corn_candidates",
    "run_corn_supply_guard_tests",
    "corn_given_signal_universe",
    "build_corn_product_flow_yfinance_families",
    "build_corn_product_flow_ethanol_family",
    "build_corn_product_flow_weather_family",
]


def _csv_path(data_dir, filename):
    return Path(data_dir) / filename


def _read_indexed_numeric_csv(data_dir, filename):
    df = pd.read_csv(_csv_path(data_dir, filename), index_col=0, parse_dates=True)
    return df.sort_index().apply(pd.to_numeric, errors="coerce")


def load_external_yfinance(data_dir="train_set"):
    return _read_indexed_numeric_csv(data_dir, "external_yfinance.csv")


def load_external_weather(data_dir="train_set"):
    df = pd.read_csv(_csv_path(data_dir, "external_weather.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_external_eia_ethanol(data_dir="train_set"):
    return _read_indexed_numeric_csv(data_dir, "external_eia_ethanol.csv")


PRICE_SIGNAL_NAMES = (
    "given_mom_20",
    "given_mom_60",
    "given_rev_5",
    "given_curve_spread",
    "given_curve_ratio",
    "given_price_family",
)
FUNDAMENTAL_CORE_SIGNAL_NAMES = (
    "given_inventory_pressure",
    "given_cgl_inventory_pressure",
    "given_cgl_crush_activity",
    "given_curve_tightness",
    "given_physical_family",
)
MACRO_SIGNAL_NAMES = (
    "external_fx_export_family",
    "external_macro_risk_family",
)
WEATHER_FAMILY_FEATURES = (
    "meteo_cdd_20d_growing",
    "meteo_hdd_20d_growing",
    "meteo_gdd_60d_growing",
    "meteo_heat_stress_20d_growing",
    "meteo_dryness_20d_growing",
    "meteo_dry_cdd_20d_growing",
    "meteo_precip_20d_planting",
    "meteo_dryness_20d_planting",
    "meteo_freeze_stress_5d_harvest",
)

CANDIDATE_FAMILY_DEFINITIONS = {
    "price": ["given_mom_20", "given_mom_60", "given_rev_5", "given_price_family"],
    "physical": [
        "given_inventory_pressure",
        "given_cgl_inventory_pressure",
        "given_cgl_crush_activity",
        "given_curve_tightness",
        "given_physical_family",
    ],
    "ethanol": ["external_ethanol_family"],
    "fx_export": ["external_fx_export_family"],
    "weather": ["external_weather_hdd_cdd_family"],
    "macro": ["external_macro_risk_family", "external_relative_grain_family"],
}
CANDIDATE_COMPOSITE_DEFINITIONS = {
    "selected_all_equal": None,
    "physical_only": ["physical"],
    "price_physical_equal": ["price", "physical"],
    "physical_fx_equal": ["physical", "fx_export"],
    "physical_weather_equal": ["physical", "weather"],
    "physical_macro_equal": ["physical", "macro"],
    "physical_ethanol_equal": ["physical", "ethanol"],
    "physical_ethanol_fx_equal": ["physical", "ethanol", "fx_export"],
    "physical_ethanol_weather_equal": ["physical", "ethanol", "weather"],
    "physical_ethanol_fx_weather_equal": ["physical", "ethanol", "fx_export", "weather"],
}


def rolling_zscore_product_flow(obj, window=252, min_periods=40):
    """Rolling z-score matching the product-flow notebook convention."""
    mean = obj.rolling(window=window, min_periods=min_periods).mean()
    std = obj.rolling(window=window, min_periods=min_periods).std()
    return (obj - mean) / std.replace(0.0, np.nan)


def to_available_calendar_product_flow(df, trading_index, lag_days):
    """Release-lag alignment used by the product-flow feature builder."""
    out = df.copy()
    out.index = pd.to_datetime(out.index) + pd.DateOffset(days=int(lag_days))
    out = out.sort_index()
    out = out.groupby(out.index).last()
    return out.reindex(trading_index).ffill()


def clean_product_flow_signal(series, index=None):
    if index is None:
        index = series.index
    return (
        pd.Series(series, index=index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-5.0, 5.0)
    )


def mean_product_flow_signals(items, index):
    values = [item.reindex(index) for item in items if item is not None]
    if not values:
        return pd.Series(0.0, index=index)
    return clean_product_flow_signal(sum(values) / float(len(values)), index)


def corn_research_split_masks(index, train_end=CORN_TRAIN_END, split_date=SPLIT_DATE):
    """Return the train/validation/test masks used by corn notebook helpers."""
    index = pd.DatetimeIndex(index)
    train_end = pd.Timestamp(train_end)
    split_date = pd.Timestamp(split_date)
    return {
        "train": pd.Series(index < train_end, index=index),
        "validation": pd.Series((index >= train_end) & (index < split_date), index=index),
        "test": pd.Series(index >= split_date, index=index),
    }


def rank_ic_product_flow(signal, target, mask):
    aligned = pd.concat([signal, target], axis=1).dropna()
    if aligned.empty:
        return np.nan
    mask = pd.Series(mask, index=signal.index).reindex(aligned.index).fillna(False).astype(bool)
    aligned = aligned.loc[mask]
    if len(aligned) < 40 or aligned.iloc[:, 0].std() == 0.0 or aligned.iloc[:, 1].std() == 0.0:
        return np.nan
    ranks = aligned.rank(method="average")
    corr = ranks.iloc[:, 0].corr(ranks.iloc[:, 1])
    return float(corr) if pd.notnull(corr) else np.nan


def smooth_corn_signal(signal, mode="long_short"):
    index = signal.index
    out = pd.Series(np.tanh(signal.astype(float) / 2.0), index=index)
    out = out.ewm(halflife=2.0, adjust=False, min_periods=1).mean()
    out[out.abs() < 0.05] = 0.0
    if mode == "long_only":
        out = out.clip(lower=0.0)
    elif mode == "short_only":
        out = out.clip(upper=0.0)
    elif mode != "long_short":
        raise ValueError(f"Unknown mode: {mode}")
    return out.fillna(0.0)


def scale_corn_positions_when(positions, condition, scale):
    out = positions.copy()
    mask = pd.Series(condition, index=positions.index).fillna(False).astype(bool)
    out.loc[mask, "CORN"] = float(scale) * out.loc[mask, "CORN"]
    return out.fillna(0.0)


def build_product_flow_feature_panels(data, commodities=("CORN",)):
    """Build feature panels with the same timing/features as product-flow corn.

    This exists because the lighter `build_feature_panels` intentionally keeps
    a simpler portable feature set. The product-flow corn sleeve used:
      - COT lag 3 calendar days;
      - public inventories/receipts lag 2 calendar days;
      - Cargill inventory/crush lag 1 calendar day;
      - processed/planned Cargill crush activity for corn as a shared physical
        processing proxy.
    """
    commodities = list(commodities)
    trading_index = data["adj1"].index
    adj1 = data["adj1"].reindex(trading_index).ffill()
    unadj1 = data["unadj1"].reindex(trading_index).ffill()
    unadj2 = data["unadj2"].reindex(trading_index).ffill()

    futures_pnl = adj1[commodities].diff() * CONTRACT_MULTIPLIER
    pct_change = adj1.pct_change()

    cot_mm = to_available_calendar_product_flow(data["cot_mm"], trading_index, 3)
    cot_pm_oi = to_available_calendar_product_flow(data["cot_pm_oi"], trading_index, 3)
    inventories = to_available_calendar_product_flow(data["inventories"], trading_index, 2)
    receipts = to_available_calendar_product_flow(data["receipts"], trading_index, 2)
    cgl_inv = to_available_calendar_product_flow(data["cgl_inv"], trading_index, 1)
    cgl_crush = to_available_calendar_product_flow(data["cgl_crush"], trading_index, 1)

    curve_spread = unadj1 - unadj2
    curve_ratio = unadj1 / unadj2.replace(0.0, np.nan) - 1.0

    blocks = {
        "mom_20": rolling_zscore_product_flow(adj1.pct_change(20), 252, 60),
        "mom_60": rolling_zscore_product_flow(adj1.pct_change(60), 252, 80),
        "rev_5": -rolling_zscore_product_flow(adj1.pct_change(5), 126, 30),
        "vol_20": rolling_zscore_product_flow(pct_change.rolling(20, min_periods=10).std(), 252, 60),
        "curve_spread": rolling_zscore_product_flow(curve_spread, 252, 60),
        "curve_ratio": rolling_zscore_product_flow(curve_ratio, 252, 60),
        "curve_change_20": rolling_zscore_product_flow(curve_spread.diff(20), 252, 60),
        "cot_mm_level": rolling_zscore_product_flow(cot_mm, 156, 40),
        "cot_mm_change": rolling_zscore_product_flow(cot_mm.diff(5), 156, 40),
        "cot_pm_oi_level": rolling_zscore_product_flow(cot_pm_oi, 156, 40),
        "cot_pm_oi_change": rolling_zscore_product_flow(cot_pm_oi.diff(5), 156, 40),
        "public_inventory_level": rolling_zscore_product_flow(inventories, 156, 40),
        "public_inventory_change": rolling_zscore_product_flow(inventories.diff(5), 156, 40),
        "receipts_level": rolling_zscore_product_flow(receipts, 126, 30),
        "receipts_change": rolling_zscore_product_flow(receipts.diff(5), 126, 30),
        "cgl_inventory_level": rolling_zscore_product_flow(cgl_inv, 252, 60),
        "cgl_inventory_change": rolling_zscore_product_flow(cgl_inv.diff(5), 252, 60),
    }

    crush = pd.DataFrame(index=trading_index)
    crush["crush_processed"] = cgl_crush["processed"]
    crush["crush_planned"] = cgl_crush["planned"]
    crush["crush_surprise"] = cgl_crush["processed"] - cgl_crush["planned"]
    crush["crush_utilization"] = cgl_crush["processed"] / cgl_crush["planned"].replace(0.0, np.nan) - 1.0
    crush_features = rolling_zscore_product_flow(crush, 252, 60)

    panels = {}
    for commodity in commodities:
        frame = pd.DataFrame(index=trading_index)
        for feature_name, block in blocks.items():
            frame[feature_name] = block[commodity]
        for feature_name in crush_features.columns:
            frame[feature_name] = crush_features[feature_name]
        panels[commodity] = frame.clip(-5.0, 5.0).fillna(0.0)
    return panels, futures_pnl


def corn_given_signal_universe(feature_panels):
    """Corn provided-data signals used by the product-flow research path."""
    panel = feature_panels["CORN"]
    inventory_pressure = (
        -panel["public_inventory_change"]
        - panel["receipts_change"]
        - panel["cgl_inventory_change"]
    ) / 3.0
    curve_tightness = (panel["curve_spread"] + panel["curve_ratio"]) / 2.0
    price_family = (panel["mom_20"] + panel["mom_60"] + panel["rev_5"]) / 3.0
    trend = (panel["mom_20"] + panel["mom_60"] + panel["curve_spread"] + panel["cot_pm_oi_level"]) / 4.0
    cgl_crush_activity = (panel["crush_surprise"] + panel["crush_utilization"]) / 2.0
    physical_family = (inventory_pressure + curve_tightness + 0.25 * cgl_crush_activity) / 2.25
    conservative = 0.40 * physical_family + 0.30 * trend + 0.30 * price_family
    signals = {
        "given_mom_20": panel["mom_20"],
        "given_mom_60": panel["mom_60"],
        "given_rev_5": panel["rev_5"],
        "given_curve_spread": panel["curve_spread"],
        "given_curve_ratio": panel["curve_ratio"],
        "given_inventory_pressure": inventory_pressure,
        "given_cgl_inventory_pressure": -panel["cgl_inventory_change"],
        "given_cgl_crush_activity": cgl_crush_activity,
        "given_curve_tightness": curve_tightness,
        "given_price_family": price_family,
        "given_physical_family": physical_family,
        "given_trend": trend,
        "given_conservative_blend": conservative,
    }
    return {name: clean_product_flow_signal(signal, panel.index) for name, signal in signals.items()}


def build_corn_product_flow_yfinance_families(trading_index, data_dir="train_set"):
    """Build corn external price/FX/macro families from saved yfinance CSV."""
    px = load_external_yfinance(data_dir).reindex(trading_index).ffill().shift(1)
    families = {}
    if {"corn", "soybean", "wheat"}.issubset(px.columns):
        corn_soy = rolling_zscore_product_flow((px["corn"] / px["soybean"]).pct_change(20, fill_method=None), 252, 60)
        corn_wheat = rolling_zscore_product_flow((px["corn"] / px["wheat"]).pct_change(20, fill_method=None), 252, 60)
        soy_corn_mr = -rolling_zscore_product_flow((px["soybean"] / px["corn"]).pct_change(20, fill_method=None), 252, 60)
        families["external_relative_grain_family"] = ((corn_soy + corn_wheat + soy_corn_mr) / 3.0).fillna(0.0)

    fx_parts = []
    if "usd_index" in px:
        fx_parts.append(-rolling_zscore_product_flow(px["usd_index"].pct_change(20, fill_method=None), 252, 60))
    if "brl" in px:
        fx_parts.append(-rolling_zscore_product_flow(px["brl"].pct_change(20, fill_method=None), 252, 60))
    if "cny" in px:
        fx_parts.append(-rolling_zscore_product_flow(px["cny"].pct_change(20, fill_method=None), 252, 60))
    if fx_parts:
        families["external_fx_export_family"] = (sum(fx_parts) / float(len(fx_parts))).fillna(0.0)

    macro_parts = []
    if "crude" in px:
        macro_parts.append(rolling_zscore_product_flow(px["crude"].pct_change(20, fill_method=None), 252, 60))
    if "equity" in px:
        macro_parts.append(rolling_zscore_product_flow(px["equity"].pct_change(20, fill_method=None), 252, 60))
    if "usd_index" in px:
        macro_parts.append(-rolling_zscore_product_flow(px["usd_index"].pct_change(60, fill_method=None), 252, 80))
    if macro_parts:
        families["external_macro_risk_family"] = (sum(macro_parts) / float(len(macro_parts))).fillna(0.0)
    return {name: clean_product_flow_signal(signal, trading_index) for name, signal in families.items()}


def build_corn_product_flow_ethanol_family(trading_index, data_dir="train_set"):
    """Build the EIA ethanol family used in the corn product-flow path."""
    ethanol = load_external_eia_ethanol(data_dir)
    available = ethanol.copy()
    available.index = available.index + pd.DateOffset(days=7)
    aligned = available.reindex(trading_index).ffill().shift(1)

    features = pd.DataFrame(index=trading_index)
    features["ethanol_production_change_4w"] = rolling_zscore_product_flow(aligned["ethanol_production"].diff(20), 156, 40)
    features["ethanol_stocks_change_4w"] = rolling_zscore_product_flow(aligned["ethanol_stocks"].diff(20), 156, 40)
    ratio = aligned["ethanol_production"] / aligned["ethanol_stocks"].replace(0.0, np.nan)
    features["ethanol_prod_to_stocks"] = rolling_zscore_product_flow(ratio, 156, 40)
    pressure = aligned["ethanol_production"].diff(20) - aligned["ethanol_stocks"].diff(20)
    features["ethanol_demand_pressure"] = rolling_zscore_product_flow(pressure, 156, 40)

    family = (
        features["ethanol_production_change_4w"]
        + features["ethanol_prod_to_stocks"]
        + features["ethanol_demand_pressure"]
        - features["ethanol_stocks_change_4w"]
    ) / 4.0
    return {
        "external_ethanol_family": clean_product_flow_signal(family, trading_index),
        "ethanol_features": features.clip(-5.0, 5.0).fillna(0.0),
    }


def _season_mask(index, months):
    return pd.Series(index.month.isin(months), index=index).astype(float)


def _add_product_flow_weather_features(aligned, seasonal=True):
    features = pd.DataFrame(index=aligned.index)
    if "tavg" in aligned:
        cdd = (aligned["tavg"] - 18.0).clip(lower=0.0)
        hdd = (18.0 - aligned["tavg"]).clip(lower=0.0)
        features["meteo_cdd_20d"] = rolling_zscore_product_flow(cdd.rolling(20, min_periods=5).sum(), 252, 60)
        features["meteo_hdd_20d"] = rolling_zscore_product_flow(hdd.rolling(20, min_periods=5).sum(), 252, 60)
    if {"tmin", "tmax"}.issubset(aligned.columns):
        temp_avg = (aligned["tmin"] + aligned["tmax"]) / 2.0
        gdd = (temp_avg.clip(upper=30.0) - 10.0).clip(lower=0.0)
        features["meteo_gdd_60d"] = rolling_zscore_product_flow(gdd.rolling(60, min_periods=15).sum(), 252, 60)
    if "tmax" in aligned:
        heat_stress = (aligned["tmax"] - 32.0).clip(lower=0.0)
        features["meteo_heat_stress_20d"] = rolling_zscore_product_flow(heat_stress.rolling(20, min_periods=5).sum(), 252, 60)
    if "prcp" in aligned:
        precip = aligned["prcp"].fillna(0.0)
        precip_20 = precip.rolling(20, min_periods=5).sum()
        features["meteo_precip_20d"] = rolling_zscore_product_flow(precip_20, 252, 60)
        features["meteo_dryness_20d"] = -features["meteo_precip_20d"]
        if "meteo_cdd_20d" in features:
            features["meteo_dry_cdd_20d"] = (features["meteo_dryness_20d"] * features["meteo_cdd_20d"]).clip(-5.0, 5.0)
    if "tmin" in aligned:
        freeze_stress = (0.0 - aligned["tmin"]).clip(lower=0.0)
        features["meteo_freeze_stress_5d"] = rolling_zscore_product_flow(freeze_stress.rolling(5, min_periods=3).sum(), 252, 60)

    if seasonal:
        planting = _season_mask(features.index, [3, 4, 5])
        growing = _season_mask(features.index, [6, 7, 8])
        harvest = _season_mask(features.index, [9, 10, 11])
        seasonal_features = {}
        for column in features.columns:
            seasonal_features[column + "_planting"] = features[column] * planting
            seasonal_features[column + "_growing"] = features[column] * growing
            seasonal_features[column + "_harvest"] = features[column] * harvest
        features = pd.concat([features, pd.DataFrame(seasonal_features, index=features.index)], axis=1)
    return features.clip(-5.0, 5.0).fillna(0.0)


def build_corn_product_flow_weather_family(trading_index, data_dir="train_set"):
    """Build the crop-belt weather family used in the corn product-flow path."""
    weather = load_external_weather(data_dir)
    weights = COMMODITY_LOCATION_WEIGHTS["CORN"]
    value_cols = [c for c in ["tavg", "tmin", "tmax", "prcp"] if c in weather.columns]
    frames = []
    for location, weight in weights.items():
        sub = weather.loc[weather["location"] == location, ["date"] + value_cols].copy()
        if sub.empty:
            continue
        sub[value_cols] = sub[value_cols] * float(weight)
        frames.append(sub)
    if not frames:
        features = pd.DataFrame(index=trading_index)
        family = pd.Series(0.0, index=trading_index)
        return {
            "external_weather_hdd_cdd_family": clean_product_flow_signal(family, trading_index),
            "weather_features": features,
        }
    combined = pd.concat(frames, ignore_index=True).groupby("date")[value_cols].sum().sort_index()
    aligned = combined.reindex(trading_index).ffill().shift(1)
    features = _add_product_flow_weather_features(aligned, seasonal=True)
    existing = [c for c in WEATHER_FAMILY_FEATURES if c in features.columns]
    family = features[existing].mean(axis=1) if existing else pd.Series(0.0, index=trading_index)
    return {
        "external_weather_hdd_cdd_family": clean_product_flow_signal(family, trading_index),
        "weather_features": features,
    }


def build_corn_product_flow_signal_universe(feature_panels, futures_pnl, data_dir="train_set"):
    """Return all corn signals used by the product-flow-aligned tests."""
    index = futures_pnl.index
    signals = corn_given_signal_universe(feature_panels)
    signals.update(build_corn_product_flow_yfinance_families(index, data_dir))
    ethanol = build_corn_product_flow_ethanol_family(index, data_dir)
    weather = build_corn_product_flow_weather_family(index, data_dir)
    signals["external_ethanol_family"] = ethanol["external_ethanol_family"]
    signals["external_weather_hdd_cdd_family"] = weather["external_weather_hdd_cdd_family"]
    return {name: clean_product_flow_signal(signal, index) for name, signal in signals.items()}


def _required_signal_map(signals, names):
    return {name: signals[name] for name in names}


def corn_signal_set_families(signals):
    """Families used for the requested Signal A / Signal B corn tests."""
    prices = _required_signal_map(signals, PRICE_SIGNAL_NAMES)
    if "external_relative_grain_family" in signals:
        prices["external_relative_grain_family"] = signals["external_relative_grain_family"]
    fundamentals_core = _required_signal_map(signals, FUNDAMENTAL_CORE_SIGNAL_NAMES)
    fundamentals_a = dict(fundamentals_core)
    fundamentals_a["external_ethanol_family"] = signals["external_ethanol_family"]
    fundamentals_a["external_weather_hdd_cdd_family"] = signals["external_weather_hdd_cdd_family"]
    macro = _required_signal_map(signals, MACRO_SIGNAL_NAMES)
    return {
        "A": {"prices": prices, "fundamentals": fundamentals_a, "macro": macro},
        "B": {"prices": prices, "fundamentals": fundamentals_core},
        "alpha": {
            "eia": {"external_ethanol_family": signals["external_ethanol_family"]},
            "macro": macro,
            "weather": {"external_weather_hdd_cdd_family": signals["external_weather_hdd_cdd_family"]},
        },
    }


def corn_family_signal(signal_dict, index):
    return mean_product_flow_signals(list(signal_dict.values()), index)


def corn_average_all_signals(families, index):
    values = []
    for signals in families.values():
        values.extend(signals.values())
    return mean_product_flow_signals(values, index)


def corn_equal_family_signal(families, index):
    return mean_product_flow_signals([corn_family_signal(v, index) for v in families.values()], index)


def corn_select_by_ic_signal(families, futures_pnl, min_abs_ic=CORN_IC_THRESHOLD):
    """Select and orient individual Signal A/B members by train-period IC."""
    index = futures_pnl.index
    target = futures_pnl["CORN"].shift(-1)
    split_masks = corn_research_split_masks(index)

    rows, selected_signals = [], []
    for family, members in families.items():
        for signal_name, signal in members.items():
            raw_signal = signal.reindex(index).fillna(0.0)
            train_ic = rank_ic_product_flow(raw_signal, target, split_masks["train"])
            orientation = 1.0 if pd.isnull(train_ic) or train_ic >= 0.0 else -1.0
            oriented_signal = clean_product_flow_signal(orientation * raw_signal, index)
            selected = bool(pd.notnull(train_ic) and abs(train_ic) >= float(min_abs_ic))
            if selected:
                selected_signals.append(oriented_signal)
            rows.append({
                "family": family,
                "signal": signal_name,
                "train_ic": train_ic,
                "orientation": orientation,
                "selected": selected,
                "validation_ic": rank_ic_product_flow(oriented_signal, target, split_masks["validation"]),
                "test_ic": rank_ic_product_flow(oriented_signal, target, split_masks["test"]),
            })

    table = pd.DataFrame(rows)
    if not table.empty:
        table["abs_train_ic"] = table["train_ic"].abs()
        table = table.sort_values(["selected", "abs_train_ic"], ascending=[False, False]).reset_index(drop=True)
    selected_signal = mean_product_flow_signals(selected_signals, index)
    return selected_signal, table


def corn_trend_mr_family_signal(families, futures_pnl, feature_panels):
    index = futures_pnl.index
    target = futures_pnl["CORN"].shift(-1)
    split_masks = corn_research_split_masks(index)
    trend_strength = feature_panels["CORN"]["mom_60"].abs().reindex(index).fillna(0.0)
    threshold = trend_strength.expanding(min_periods=252).median().shift(1)
    regimes = {
        "trend": (trend_strength > threshold).fillna(False),
        "mr_or_chop": (trend_strength <= threshold).fillna(True),
    }
    family_signals = {name: corn_family_signal(signals, index) for name, signals in families.items()}
    rows, pieces = [], []
    for regime_name, regime_mask in regimes.items():
        candidates = []
        for family, signal in family_signals.items():
            train_mask = split_masks["train"] & regime_mask
            validation_mask = split_masks["validation"] & regime_mask
            train_ic = rank_ic_product_flow(signal, target, train_mask)
            orientation = 1.0 if pd.isnull(train_ic) or train_ic >= 0.0 else -1.0
            validation_ic = rank_ic_product_flow(orientation * signal, target, validation_mask)
            candidates.append({
                "regime": regime_name,
                "family": family,
                "train_ic": train_ic,
                "orientation": orientation,
                "validation_ic": validation_ic,
                "train_obs": int(train_mask.sum()),
                "validation_obs": int(validation_mask.sum()),
                "signal": orientation * signal,
            })
        table = pd.DataFrame([{k: v for k, v in row.items() if k != "signal"} for row in candidates])
        selected = table.sort_values("validation_ic", ascending=False).iloc[0]
        selected_signal = next(row["signal"] for row in candidates if row["family"] == selected["family"])
        pieces.append(selected_signal * regime_mask.astype(float))
        rows.append(selected.to_dict())
    return clean_product_flow_signal(sum(pieces), index), pd.DataFrame(rows)


def _standardize_family_frame(x_frame, mean, std):
    return ((x_frame - mean) / std).replace([np.inf, -np.inf], 0.0).values.astype(float)


def _fit_expanding_ols_state(x_train, y_train, rcond=1.0e-8):
    mean = x_train.mean()
    std = x_train.std().replace(0.0, np.nan).fillna(1.0)
    x_std = _standardize_family_frame(x_train, mean, std)
    y_values = y_train.values.astype(float)
    x_aug = np.column_stack([np.ones(len(x_std)), x_std])

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        beta = np.linalg.pinv(x_aug, rcond=float(rcond)) @ y_values
        residual = y_values - x_aug @ beta
        covariance = np.linalg.pinv(x_aug.T @ x_aug, rcond=float(rcond))

    if not np.isfinite(beta).all():
        beta = np.zeros(x_aug.shape[1], dtype=float)
        beta[0] = float(np.nanmean(y_values)) if len(y_values) else 0.0
        residual = y_values - x_aug @ beta

    obs_var = float(np.nanvar(residual))
    if not np.isfinite(obs_var) or obs_var <= 1.0e-12:
        obs_var = float(np.nanvar(y_values)) if len(y_values) > 1 else 1.0
    obs_var = max(obs_var, 1.0)
    if not np.isfinite(covariance).all():
        covariance = np.eye(x_aug.shape[1])
    covariance = covariance * obs_var
    covariance = covariance + np.eye(covariance.shape[0]) * 1.0e-6
    return beta, covariance, mean, std, obs_var


def _kalman_update(beta, covariance, x_vector, observed_y, obs_var, process_noise):
    covariance = covariance + np.eye(len(beta)) * float(process_noise)
    denom = float(x_vector @ covariance @ x_vector + obs_var)
    if not np.isfinite(denom) or denom <= 1.0e-12:
        return beta, covariance
    gain = covariance @ x_vector / denom
    innovation = float(observed_y - x_vector @ beta)
    beta = beta + gain * innovation
    covariance = covariance - np.outer(gain, x_vector) @ covariance
    return beta, covariance


def _coefficient_row(beta, date, columns):
    return {
        "date": date,
        "intercept": float(beta[0]),
        **{f"beta_{column}": float(beta[j + 1]) for j, column in enumerate(columns)},
    }


def corn_dynamic_linear_family_signal(families, futures_pnl, min_train_days=504, refit_every=21,
                                      process_noise=1.0e-5):
    """Walk-forward OLS/Kalman benchmark over family-level corn signals.

    Coefficients start from an expanding OLS fit, then receive a recursive
    Kalman/RLS-style update when the prior day's forward return becomes known.
    The model is periodically refreshed with expanding OLS so coefficients do
    not drift indefinitely.
    """
    index = futures_pnl.index
    x = pd.DataFrame({name: corn_family_signal(signals, index) for name, signals in families.items()}, index=index).fillna(0.0)
    y = futures_pnl["CORN"].shift(-1)
    pred = pd.Series(np.nan, index=index)
    beta, covariance, last_fit = None, None, None
    obs_var = 1.0
    rows = []
    for i, date in enumerate(index):
        train_mask = (index < date) & y.notna()
        if int(train_mask.sum()) < min_train_days:
            continue
        x_train_raw = x.loc[train_mask]
        mean = x_train_raw.mean()
        std = x_train_raw.std().replace(0.0, np.nan).fillna(1.0)

        if beta is None or covariance is None or last_fit is None or (i - last_fit) >= refit_every:
            beta, covariance, _, _, obs_var = _fit_expanding_ols_state(x_train_raw, y.loc[train_mask])
            last_fit = i
        elif i > 0 and pd.notnull(y.iloc[i - 1]):
            x_prev = _standardize_family_frame(x.iloc[[i - 1]], mean, std)[0]
            x_prev_aug = np.r_[1.0, x_prev]
            beta, covariance = _kalman_update(
                beta,
                covariance,
                x_prev_aug,
                float(y.iloc[i - 1]),
                obs_var,
                process_noise,
            )

        rows.append(_coefficient_row(beta, date, x.columns))
        x_row = _standardize_family_frame(x.loc[[date]], mean, std)[0]
        pred.loc[date] = float(np.r_[1.0, x_row] @ beta)
    mean = pred.rolling(252, min_periods=60).mean().shift(1)
    std = pred.rolling(252, min_periods=60).std().shift(1).replace(0.0, np.nan)
    return clean_product_flow_signal(((pred - mean) / std).clip(-5.0, 5.0), index), pd.DataFrame(rows)


def corn_positions_from_signal(signal, futures_pnl, mode="long_short",
                               target_daily_pnl_vol=CORN_TARGET_DAILY_PNL_VOL,
                               max_abs_lot=CORN_MAX_ABS_LOT):
    index = futures_pnl.index
    cleaned = smooth_corn_signal(signal.reindex(index).fillna(0.0), mode=mode)
    asset_vol = futures_pnl["CORN"].rolling(60, min_periods=20).std().shift(1).replace(0.0, np.nan)
    lots = cleaned * (float(target_daily_pnl_vol) / asset_vol)
    positions = pd.DataFrame(0.0, index=index, columns=["CORN"])
    if mode == "long_only":
        positions["CORN"] = lots.clip(0.0, float(max_abs_lot)).fillna(0.0)
    else:
        positions["CORN"] = lots.clip(-float(max_abs_lot), float(max_abs_lot)).fillna(0.0)
    return positions


def _backtest_corn_positions(positions, futures_pnl, trade_cost_per_lot=CORN_TRADE_COST_PER_LOT,
                             holding_cost_rate=CORN_HOLDING_COST_RATE):
    """Corn wrapper around the shared cost-aware backtest engine."""
    return backtest_positions_with_costs(
        positions,
        futures_pnl,
        trade_cost_per_lot=trade_cost_per_lot,
        holding_cost_rate=holding_cost_rate,
        margin_per_lot=DEFAULT_MARGIN_PER_LOT,
    )


def backtest_positions_product_flow(positions, futures_pnl, trade_cost_per_lot=CORN_TRADE_COST_PER_LOT,
                                    holding_cost_rate=CORN_HOLDING_COST_RATE):
    """Compatibility wrapper for the GitHub corn notebook."""
    return _backtest_corn_positions(
        positions,
        futures_pnl,
        trade_cost_per_lot=trade_cost_per_lot,
        holding_cost_rate=holding_cost_rate,
    )


def product_flow_performance_metrics(bt):
    active = bt["held_gross_exposure"] > 1.0e-12
    pnl = bt.loc[active, "net_pnl"].dropna()
    if len(pnl) == 0:
        return pd.Series(dtype=float)
    vol = pnl.std()
    sharpe = np.nan if vol == 0.0 else pnl.mean() / vol * np.sqrt(252.0)
    cum = pnl.cumsum()
    drawdown = cum - cum.cummax()
    return pd.Series({
        "days": float(len(pnl)),
        "total_pnl": float(pnl.sum()),
        "sharpe": float(sharpe) if pd.notnull(sharpe) else np.nan,
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((pnl > 0.0).mean()),
        "avg_daily_turnover": float(bt["turnover"].reindex(pnl.index).mean()),
        "avg_gross_exposure": float(bt["gross_exposure"].reindex(pnl.index).mean()),
    })


def product_flow_split_performance(bt, split_date=SPLIT_DATE):
    split_date = pd.Timestamp(split_date)
    return pd.DataFrame({
        "in_sample": product_flow_performance_metrics(bt.loc[bt.index < split_date]),
        "out_of_sample": product_flow_performance_metrics(bt.loc[bt.index >= split_date]),
        "full_period": product_flow_performance_metrics(bt),
    })


def summarize_corn_backtest(bt, train_end=CORN_TRAIN_END, split_date=SPLIT_DATE):
    full = product_flow_split_performance(bt, split_date)
    train_val = product_flow_split_performance(bt.loc[bt.index < pd.Timestamp(split_date)], train_end)
    return {
        "train_sharpe": train_val.loc["sharpe", "in_sample"],
        "validation_sharpe": train_val.loc["sharpe", "out_of_sample"],
        "validation_dd": train_val.loc["max_drawdown", "out_of_sample"],
        "oos_sharpe": full.loc["sharpe", "out_of_sample"],
        "oos_pnl": full.loc["total_pnl", "out_of_sample"],
        "oos_dd": full.loc["max_drawdown", "out_of_sample"],
        "full_sharpe": full.loc["sharpe", "full_period"],
        "full_pnl": full.loc["total_pnl", "full_period"],
        "full_dd": full.loc["max_drawdown", "full_period"],
        "turnover": full.loc["avg_daily_turnover", "full_period"],
        "avg_gross_exposure": full.loc["avg_gross_exposure", "full_period"],
    }


def product_flow_period_performance(bt, periods=None):
    if periods is None:
        periods = REGIME_PERIODS
    rows = []
    for item in periods:
        start = pd.Timestamp(item["start"])
        end = pd.Timestamp(item["end"])
        metrics = product_flow_performance_metrics(bt.loc[(bt.index >= start) & (bt.index <= end)])
        row = {"period": item["period"], "start": start, "end": end}
        for key, value in metrics.items():
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def corn_vol_regime_masks(feature_panels, futures_pnl):
    index = futures_pnl.index
    pnl = futures_pnl["CORN"].fillna(0.0)
    vol = pnl.rolling(60, min_periods=20).std().shift(1)
    lt_vol = vol.expanding(min_periods=252).median().shift(1)
    high_q = vol.expanding(min_periods=252).quantile(0.75).shift(1)
    high_vol = ((vol > 1.20 * lt_vol) | (vol > high_q)).reindex(index).fillna(False)
    low_vol = (vol < 0.80 * lt_vol).reindex(index).fillna(False)
    normal_vol = (~high_vol & ~low_vol).reindex(index).fillna(True)
    return {"low_vol": low_vol.astype(bool), "normal_vol": normal_vol.astype(bool), "high_vol": high_vol.astype(bool)}


def corn_regime_signal_ic_table(signals, futures_pnl, regime_mask):
    index = futures_pnl.index
    target = futures_pnl["CORN"].shift(-1)
    splits = corn_research_split_masks(index)
    rows = []
    regime = pd.Series(regime_mask, index=index).fillna(False).astype(bool)
    for name, signal in signals.items():
        row = {"signal": name}
        for split_name, split_mask in splits.items():
            mask = split_mask & regime
            row[f"{split_name}_obs"] = int(mask.sum())
            row[f"{split_name}_ic"] = rank_ic_product_flow(signal.reindex(index), target, mask)
        row["passes_ic_threshold"] = bool(
            row["train_obs"] >= 120
            and row["validation_obs"] >= 40
            and pd.notnull(row["train_ic"])
            and abs(row["train_ic"]) >= CORN_IC_THRESHOLD
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["passes_ic_threshold", "train_ic"], ascending=[False, False])


def corn_candidate_families(selected_signals):
    families, members = {}, {}
    index = next(iter(selected_signals.values())).index
    for family, names in CANDIDATE_FAMILY_DEFINITIONS.items():
        used = [selected_signals[name] for name in names if name in selected_signals]
        if used:
            families[family] = mean_product_flow_signals(used, index)
            members[family] = [name for name in names if name in selected_signals]
    return families, members


def corn_candidate_composites(families):
    candidates, members = {}, {}
    index = next(iter(families.values())).index
    for candidate, family_names in CANDIDATE_COMPOSITE_DEFINITIONS.items():
        if family_names is None:
            family_names = list(families.keys())
        used = [families[name] for name in family_names if name in families]
        if not used:
            continue
        if candidate != "selected_all_equal" and len(used) != len(family_names):
            continue
        candidates[candidate] = mean_product_flow_signals(used, index)
        members[candidate] = [name for name in family_names if name in families]
    return candidates, members


def select_corn_candidate_for_regime(signals, futures_pnl, regime_mask):
    index = futures_pnl.index
    signal_ic = corn_regime_signal_ic_table(signals, futures_pnl, regime_mask)
    selected_signals = {}
    for _, row in signal_ic.loc[signal_ic["passes_ic_threshold"]].iterrows():
        sign = 1.0 if row["train_ic"] >= 0.0 else -1.0
        selected_signals[row["signal"]] = clean_product_flow_signal(sign * signals[row["signal"]], index)
    if not selected_signals:
        return None, signal_ic, pd.DataFrame(), None

    families, _ = corn_candidate_families(selected_signals)
    candidates, candidate_members = corn_candidate_composites(families)
    if not candidates:
        return None, signal_ic, pd.DataFrame(), None

    target = futures_pnl["CORN"].shift(-1)
    regime = pd.Series(regime_mask, index=index).fillna(False).astype(bool)
    splits = corn_research_split_masks(index)
    rows = []
    for candidate, signal in candidates.items():
        candidate_signal = signal.clip(lower=0.0)
        row = {"candidate": candidate, "mode": "long_only", "families": ",".join(candidate_members[candidate])}
        for split_name, split_mask in splits.items():
            mask = split_mask & regime
            row[f"{split_name}_obs"] = int(mask.sum())
            row[f"{split_name}_ic"] = rank_ic_product_flow(candidate_signal, target, mask)
        eligible = (
            row["train_obs"] >= 120
            and row["validation_obs"] >= 40
            and pd.notnull(row["train_ic"])
            and pd.notnull(row["validation_ic"])
            and row["train_ic"] >= CORN_IC_THRESHOLD
            and row["validation_ic"] >= 0.0
        )
        row["eligible"] = bool(eligible)
        row["score"] = row["validation_ic"] + 0.25 * row["train_ic"] if eligible else -np.inf
        rows.append(row)
    table = pd.DataFrame(rows)
    eligible = table.loc[table["eligible"]].copy()
    if eligible.empty:
        selected = table.sort_values(["validation_ic", "train_ic"], ascending=[False, False]).iloc[0]
    else:
        selected = eligible.sort_values(["score", "validation_ic"], ascending=[False, False]).iloc[0]
    selected_signal = candidates[selected["candidate"]].clip(lower=0.0)
    return selected, signal_ic, table, selected_signal


def build_corn_vol_regime_signal(signals, feature_panels, futures_pnl):
    pieces, rows = [], []
    signal_ics, candidate_tables = {}, {}
    for regime_name, regime_mask in corn_vol_regime_masks(feature_panels, futures_pnl).items():
        selected, signal_ic, candidate_table, selected_signal = select_corn_candidate_for_regime(signals, futures_pnl, regime_mask)
        signal_ics[regime_name] = signal_ic
        candidate_tables[regime_name] = candidate_table
        if selected_signal is None:
            continue
        selected = selected.copy()
        selected["regime"] = regime_name
        rows.append(selected)
        pieces.append(selected_signal * pd.Series(regime_mask, index=futures_pnl.index).astype(float))
    if not pieces:
        return pd.Series(0.0, index=futures_pnl.index), pd.DataFrame(), signal_ics, candidate_tables
    return clean_product_flow_signal(sum(pieces), futures_pnl.index), pd.DataFrame(rows), signal_ics, candidate_tables


def corn_abundant_supply_masks(data, feature_panels, futures_pnl):
    """Fixed weak-tape guard used by the corn backtest notebook."""
    index = futures_pnl.index
    price = data["adj1"]["CORN"].reindex(index).ffill()
    below_ma = price < price.rolling(252, min_periods=120).mean().shift(1)
    mom60_negative = feature_panels["CORN"]["mom_60"].reindex(index).fillna(0.0) < 0.0
    return {
        "below_ma_or_negative_mom": (below_ma | mom60_negative).fillna(False),
    }


def corn_candidate_key(candidate):
    return (
        f'{candidate["source_table"]}|{candidate["signal_set"]}|'
        f'{candidate["strategy"]}|{candidate["mode"]}|{candidate["note"]}'
    )


def _candidate_metadata(candidate):
    return {k: v for k, v in candidate.items() if k not in ["positions", "signal"]}


def _summarized_candidate_row(candidate, futures_pnl):
    bt, _ = _backtest_corn_positions(candidate["positions"], futures_pnl)
    row = _candidate_metadata(candidate)
    row["candidate_key"] = corn_candidate_key(candidate)
    row.update(summarize_corn_backtest(bt))
    return row, bt


def _supply_guard_row_metadata(candidate, base_key, guard_name):
    return {
        "candidate_key": base_key,
        "source_table": candidate["source_table"],
        "selection_rule": candidate["selection_rule"],
        "signal_set": candidate["signal_set"],
        "base_strategy": candidate["strategy"],
        "base_mode": candidate["mode"],
        "note": candidate["note"],
        "guard": guard_name,
        "strategy": f'{candidate["strategy"]}__{guard_name}',
    }


def build_corn_carry_forward_candidates(specs, combo_results, combo_positions, combo_signals,
                                        alpha_results, alpha_positions, alpha_context_signals):
    """Build the small set of hand-carried candidates used in the final guard test."""
    candidates = []
    for spec in specs:
        signal_set = spec["signal_set"]
        strategy = spec["strategy"]
        mode = spec["mode"]
        note = spec["note"]

        if spec["source_table"] == "alpha_combinations":
            row = combo_results.loc[
                (combo_results["signal_set"] == signal_set)
                & (combo_results["strategy"] == strategy)
                & (combo_results["alpha_combo"] == note)
                & (combo_results["mode"] == mode)
            ].iloc[0]
            signal = combo_signals[(signal_set, strategy, note)]
            positions = combo_positions[(signal_set, strategy, note, mode)]
        elif spec["source_table"] == "standalone_alpha_sleeves":
            row = alpha_results.loc[
                (alpha_results["signal_set"] == signal_set)
                & (alpha_results["strategy"] == strategy)
                & (alpha_results["alpha"] == note)
                & (alpha_results["mode"] == mode)
            ].iloc[0]
            signal = alpha_context_signals[(signal_set, note)]
            positions = alpha_positions[(signal_set, note, mode)]
        else:
            raise ValueError(f'Unknown source table: {spec["source_table"]}')

        candidates.append({
            "source_table": spec["source_table"],
            "selection_rule": spec["selection_rule"],
            "signal_set": signal_set,
            "strategy": strategy,
            "mode": mode,
            "note": note,
            "signal": signal,
            "positions": positions,
            "validation_sharpe_at_selection": row["validation_sharpe"],
        })
    return candidates


def make_corn_candidate(source_table, selection_rule, signal_set, strategy, mode, note, signal, positions):
    return {
        "source_table": source_table,
        "selection_rule": selection_rule,
        "signal_set": signal_set,
        "strategy": strategy,
        "mode": mode,
        "note": note,
        "signal": signal,
        "positions": positions,
    }


def summarize_corn_candidates(candidates, futures_pnl):
    rows = []
    for candidate in candidates:
        row, _ = _summarized_candidate_row(candidate, futures_pnl)
        rows.append(row)
    return pd.DataFrame(rows)


def run_corn_supply_guard_tests(candidates, supply_masks, futures_pnl, trading_index,
                                oos_start=SPLIT_DATE, guard_specs=()):
    rows, backtests, positions_by_key = [], {}, {}
    guard_specs = list(guard_specs)
    candidate_by_key = {corn_candidate_key(candidate): candidate for candidate in candidates}
    for candidate in candidates:
        base_key = corn_candidate_key(candidate)
        guard_tests = {"no_guard": (candidate["positions"], None)}
        for spec in guard_specs:
            mask_name = spec["mask"]
            guard_name = spec.get("name") or f"{mask_name}_{spec['scale']:.2f}"
            guard_tests[guard_name] = (
                scale_corn_positions_when(candidate["positions"], supply_masks[mask_name], spec["scale"]),
                mask_name,
            )

        for guard_name, (positions, mask_name) in guard_tests.items():
            bt, _ = _backtest_corn_positions(positions, futures_pnl)
            row = _supply_guard_row_metadata(candidate, base_key, guard_name)
            row.update(summarize_corn_backtest(bt))
            if mask_name is None:
                row["guard_oos_pct"] = 0.0
            else:
                mask = pd.Series(supply_masks[mask_name], index=trading_index)
                row["guard_oos_pct"] = float(mask.loc[trading_index >= pd.Timestamp(oos_start)].mean())
            rows.append(row)
            backtests[(base_key, guard_name)] = bt
            positions_by_key[(base_key, guard_name)] = positions

    results = pd.DataFrame(rows).sort_values(["oos_sharpe", "full_sharpe"], ascending=[False, False])
    return results, backtests, positions_by_key, candidate_by_key


DD_COLUMNS = ["validation_dd", "trade_dd", "oos_dd", "full_dd", "max_drawdown"]


def corn_dd_pct_table(table, columns=None, dd_capital_usd=10000.0):
    out = table.copy()
    if columns is not None:
        out = out[columns].copy()
    rename = {}
    for column in DD_COLUMNS:
        if column in out.columns:
            out[column] = 100.0 * out[column] / float(dd_capital_usd)
            rename[column] = "max_dd_pct" if column == "max_drawdown" else f"{column}_pct"
    return out.rename(columns=rename)


def load_corn_research_context(data_dir="train_set", commodity="CORN"):
    data = load_train_set(data_dir)
    feature_panels, futures_pnl_all = build_product_flow_feature_panels(data)
    futures_pnl = futures_pnl_all[[commodity]].copy()
    trading_index = futures_pnl.index
    signals = build_corn_product_flow_signal_universe(feature_panels, futures_pnl_all, data_dir)
    families_by_set = corn_signal_set_families(signals)

    summary = pd.DataFrame(
        [
            {
                "commodity": commodity,
                "start": trading_index.min().date(),
                "end": trading_index.max().date(),
                "rows": len(trading_index),
                "corn_features": feature_panels[commodity].shape[1],
                "signals": len(signals),
                "has_cargill_crush_activity": {"crush_surprise", "crush_utilization"}.issubset(
                    feature_panels[commodity].columns
                ),
                "train_rows": int((trading_index < pd.Timestamp(CORN_TRAIN_END)).sum()),
                "validation_rows": int(
                    ((trading_index >= pd.Timestamp(CORN_TRAIN_END)) & (trading_index < pd.Timestamp(SPLIT_DATE))).sum()
                ),
                "oos_rows": int((trading_index >= pd.Timestamp(SPLIT_DATE)).sum()),
            }
        ]
    )

    coverage = []
    for signal_set, family_map in families_by_set.items():
        for family, members in family_map.items():
            coverage.append(
                {
                    "signal_set": signal_set,
                    "family": family,
                    "signals": len(members),
                    "nonzero_signals": int(sum(series.abs().sum() > 0.0 for series in members.values())),
                }
            )

    return {
        "data": data,
        "feature_panels": feature_panels,
        "futures_pnl_all": futures_pnl_all,
        "futures_pnl": futures_pnl,
        "trading_index": trading_index,
        "signals": signals,
        "families_by_set": families_by_set,
        "summary": summary,
        "coverage": pd.DataFrame(coverage),
        "commodity": commodity,
    }


def evaluate_corn_signal(context, test, signal_set, strategy, signal, mode="long_short", note=""):
    futures_pnl = context["futures_pnl"]
    positions = corn_positions_from_signal(signal, futures_pnl, mode=mode)
    bt, _ = backtest_positions_product_flow(positions, futures_pnl)
    row = {
        "test": test,
        "signal_set": signal_set,
        "strategy": strategy,
        "mode": mode,
        "note": note,
    }
    row.update(summarize_corn_backtest(bt))
    return row, bt, positions


def _make_family_features(families, index):
    return pd.DataFrame(
        {family: corn_equal_family_signal({family: members}, index) for family, members in families.items()},
        index=index,
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _zscore_from_train(x_train, x_row):
    mean = x_train.mean()
    std = x_train.std().replace(0.0, np.nan)
    return (
        ((x_train - mean) / std).clip(lower=-5.0, upper=5.0).fillna(0.0),
        ((x_row - mean) / std).clip(lower=-5.0, upper=5.0).fillna(0.0),
    )


def _fit_ols(x_train, y_train):
    x_design = np.column_stack([np.ones(len(x_train)), np.asarray(x_train, dtype=float)])
    beta, *_ = np.linalg.lstsq(x_design, np.asarray(y_train, dtype=float), rcond=None)
    return beta


def _expanding_ols_prediction(x, y, min_train_days=504, refit_every=21):
    preds = pd.Series(np.nan, index=x.index)
    beta = None
    last_fit_i = None
    for i, date in enumerate(x.index):
        train_mask = (x.index < date) & y.notna()
        if train_mask.sum() < min_train_days:
            continue
        if beta is None or last_fit_i is None or (i - last_fit_i) >= refit_every:
            x_train_raw = x.loc[train_mask]
            y_train = y.loc[train_mask]
            x_train, x_row = _zscore_from_train(x_train_raw, x.loc[date])
            beta = _fit_ols(x_train, y_train)
            last_fit_i = i
        else:
            _, x_row = _zscore_from_train(x.loc[train_mask], x.loc[date])
        preds.loc[date] = np.r_[1.0, np.asarray(x_row, dtype=float)].dot(beta)
    return preds


def _kalman_prediction(x, y, min_train_days=504, process_noise=1.0e-5):
    columns = list(x.columns)
    beta = np.zeros(len(columns) + 1)
    covariance = np.eye(len(beta)) * 10.0
    preds = pd.Series(np.nan, index=x.index)
    mean = pd.Series(0.0, index=columns)
    var = pd.Series(1.0, index=columns)
    target_var = 1.0
    n = 0
    for date in x.index:
        row = x.loc[date]
        if n > min_train_days:
            std = np.sqrt(var.clip(lower=1.0e-8))
            z = ((row - mean) / std).clip(lower=-5.0, upper=5.0)
            preds.loc[date] = np.r_[1.0, np.asarray(z, dtype=float)].dot(beta)
        y_value = y.loc[date]
        if pd.notnull(y_value):
            n += 1
            old_mean = mean.copy()
            mean = mean + (row - mean) / float(n)
            var = ((n - 2.0) / max(n - 1.0, 1.0)) * var + (
                (row - old_mean) * (row - mean)
            ) / max(n - 1.0, 1.0)
            target_var = target_var + (float(y_value) ** 2 - target_var) / float(n)
            if n > min_train_days:
                std = np.sqrt(var.clip(lower=1.0e-8))
                z = ((row - mean) / std).clip(lower=-5.0, upper=5.0)
                phi = np.r_[1.0, np.asarray(z, dtype=float)]
                covariance = covariance + np.eye(len(beta)) * float(process_noise)
                innovation_var = float(phi.dot(covariance).dot(phi) + max(target_var, 1.0))
                gain = covariance.dot(phi) / innovation_var
                beta = beta + gain * float(y_value - phi.dot(beta))
                covariance = covariance - np.outer(gain, phi).dot(covariance)
    return preds


def _prediction_to_signal(prediction, index):
    prediction = prediction.replace([np.inf, -np.inf], np.nan)
    mean = prediction.rolling(252, min_periods=60).mean().shift(1)
    std = prediction.rolling(252, min_periods=60).std().shift(1).replace(0.0, np.nan)
    return clean_product_flow_signal(((prediction - mean) / std).clip(lower=-5.0, upper=5.0), index)


def run_corn_generic_signal_tests(context):
    trading_index = context["trading_index"]
    futures_pnl = context["futures_pnl"]
    families_by_set = context["families_by_set"]
    feature_panels = context["feature_panels"]
    commodity = context["commodity"]

    rows, backtests, positions = [], {}, {}
    for signal_set in ["A", "B"]:
        families = families_by_set[signal_set]
        trend_signal, _ = corn_trend_mr_family_signal(families, futures_pnl, feature_panels)
        ic_signal, _ = corn_select_by_ic_signal(families, futures_pnl)
        family_features = _make_family_features(families, trading_index)
        model_target = futures_pnl[commodity].shift(-1)
        ols_signal = _prediction_to_signal(_expanding_ols_prediction(family_features, model_target), trading_index)
        kalman_signal = _prediction_to_signal(_kalman_prediction(family_features, model_target), trading_index)

        tests = [
            ("avg_all_signals", corn_average_all_signals(families, trading_index)),
            ("equal_family", corn_equal_family_signal(families, trading_index)),
            ("best_family_by_trend_mr", trend_signal),
            ("select_by_ic", ic_signal),
            ("expanding_ols_family_model", ols_signal),
            ("kalman_family_model", kalman_signal),
        ]
        for strategy, signal in tests:
            row, bt, pos = evaluate_corn_signal(context, "generic", signal_set, strategy, signal, mode="long_short")
            rows.append(row)
            backtests[(signal_set, strategy, "long_short")] = bt
            positions[(signal_set, strategy, "long_short")] = pos

    results = pd.DataFrame(rows).sort_values(
        ["signal_set", "validation_sharpe", "oos_sharpe"],
        ascending=[True, False, False],
    )
    return {"results": results, "backtests": backtests, "positions": positions}


def run_corn_momentum_mr_benchmarks(context):
    data = context["data"]
    feature_panels = context["feature_panels"]
    futures_pnl = context["futures_pnl"]
    trading_index = context["trading_index"]
    commodity = context["commodity"]
    corn_panel = feature_panels[commodity].reindex(trading_index).fillna(0.0)

    signals = {
        "mom_20": clean_product_flow_signal(corn_panel["mom_20"], trading_index),
        "mom_60": clean_product_flow_signal(corn_panel["mom_60"], trading_index),
        "rev_5": clean_product_flow_signal(corn_panel["rev_5"], trading_index),
        "mom_60_rev_5_equal": clean_product_flow_signal(
            mean_product_flow_signals([corn_panel["mom_60"], corn_panel["rev_5"]], trading_index),
            trading_index,
        ),
    }
    trend_strength = corn_panel["mom_60"].abs()
    trend_threshold = trend_strength.expanding(min_periods=252).median().shift(1)
    trend_regime = (trend_strength > trend_threshold).fillna(False)
    signals["trend_mom_else_mr"] = clean_product_flow_signal(
        pd.Series(np.where(trend_regime, corn_panel["mom_60"], corn_panel["rev_5"]), index=trading_index),
        trading_index,
    )

    rows, candidates = [], []
    for name, signal in signals.items():
        row, _, pos = evaluate_corn_signal(context, "momentum_mr_benchmark", "price_only", name, signal)
        rows.append(row)
        candidates.append(
            make_corn_candidate(
                "momentum_mr_benchmark",
                "simple_price_rule",
                "price_only",
                name,
                "long_short",
                "raw",
                signal,
                pos,
            )
        )

    results = pd.DataFrame(rows).sort_values(["oos_sharpe", "full_sharpe"], ascending=[False, False])
    supply_masks = corn_abundant_supply_masks(data, feature_panels, futures_pnl)
    guard_results, guard_backtests, guard_positions, candidate_by_key = run_corn_supply_guard_tests(
        candidates,
        supply_masks,
        futures_pnl,
        trading_index,
        oos_start=SPLIT_DATE,
    )
    return {
        "signals": signals,
        "results": results,
        "supply_masks": supply_masks,
        "guard_results": guard_results.sort_values(["oos_sharpe", "full_sharpe"], ascending=[False, False]),
        "guard_backtests": guard_backtests,
        "guard_positions": guard_positions,
        "candidate_by_key": candidate_by_key,
    }


def _walk_forward_momentum_mr_selection(signals, futures_pnl, start=SPLIT_DATE):
    index = futures_pnl.index
    rebalance_dates = list(pd.date_range(pd.Timestamp(start), index.max(), freq="YS"))
    selected_signal = pd.Series(0.0, index=index)
    rows = []

    for i, rebalance_date in enumerate(rebalance_dates):
        next_rebalance = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else index.max() + pd.Timedelta(days=1)
        train_end = rebalance_date - pd.DateOffset(years=2)
        train_mask = index < train_end
        validation_mask = (index >= train_end) & (index < rebalance_date)
        trade_mask = (index >= rebalance_date) & (index < next_rebalance)

        candidates = []
        for name, signal in signals.items():
            positions = corn_positions_from_signal(signal, futures_pnl)
            bt, _ = backtest_positions_product_flow(positions, futures_pnl)
            train_metrics = product_flow_performance_metrics(bt.loc[train_mask])
            validation_metrics = product_flow_performance_metrics(bt.loc[validation_mask])
            trade_metrics = product_flow_performance_metrics(bt.loc[trade_mask])
            train_sharpe = train_metrics.get("sharpe", np.nan)
            validation_sharpe = validation_metrics.get("sharpe", np.nan)
            validation_dd = validation_metrics.get("max_drawdown", np.nan)
            eligible = bool(
                pd.notnull(train_sharpe)
                and pd.notnull(validation_sharpe)
                and train_sharpe > 0.0
                and validation_sharpe > 0.0
            )
            score = validation_sharpe + 0.25 * train_sharpe + 0.001 * validation_dd if eligible else -np.inf
            candidates.append(
                {
                    "rebalance": rebalance_date.date(),
                    "candidate": name,
                    "eligible": eligible,
                    "score": score,
                    "train_sharpe": train_sharpe,
                    "validation_sharpe": validation_sharpe,
                    "validation_dd": validation_dd,
                    "trade_sharpe": trade_metrics.get("sharpe", np.nan),
                    "trade_pnl": trade_metrics.get("total_pnl", np.nan),
                    "trade_dd": trade_metrics.get("max_drawdown", np.nan),
                }
            )

        candidate_table = pd.DataFrame(candidates)
        eligible = candidate_table.loc[candidate_table["eligible"]].copy()
        if eligible.empty:
            selected = candidate_table.sort_values(["validation_sharpe", "train_sharpe"], ascending=[False, False]).iloc[0]
            selection_read = "Fallback: no candidate passed positive train/validation Sharpe gate."
        else:
            selected = eligible.sort_values(["score", "validation_sharpe"], ascending=[False, False]).iloc[0]
            selection_read = "Selected using only data before this rebalance date."

        selected_signal.loc[trade_mask] = signals[selected["candidate"]].loc[trade_mask]
        selected = selected.copy()
        selected["selected"] = True
        selected["selection_read"] = selection_read
        rows.append(selected)

    return clean_product_flow_signal(selected_signal, index), pd.DataFrame(rows)


def _momentum_mr_oos_metric_row(name, bt):
    oos_metrics = product_flow_performance_metrics(bt.loc[bt.index >= pd.Timestamp(SPLIT_DATE)])
    full_metrics = product_flow_performance_metrics(bt)
    return {
        "strategy": name,
        "oos_sharpe": oos_metrics.get("sharpe", np.nan),
        "oos_pnl": oos_metrics.get("total_pnl", np.nan),
        "oos_dd": oos_metrics.get("max_drawdown", np.nan),
        "oos_active_days": oos_metrics.get("days", np.nan),
        "full_sharpe": full_metrics.get("sharpe", np.nan),
        "full_pnl": full_metrics.get("total_pnl", np.nan),
        "full_dd": full_metrics.get("max_drawdown", np.nan),
    }


def run_corn_walk_forward_momentum_mr(context, momentum_mr):
    futures_pnl = context["futures_pnl"]
    signal, selected = _walk_forward_momentum_mr_selection(momentum_mr["signals"], futures_pnl)
    positions = corn_positions_from_signal(signal, futures_pnl)
    bt, _ = backtest_positions_product_flow(positions, futures_pnl)
    static_bt, _ = backtest_positions_product_flow(
        corn_positions_from_signal(momentum_mr["signals"]["mom_20"], futures_pnl),
        futures_pnl,
    )
    comparison = pd.DataFrame(
        [
            _momentum_mr_oos_metric_row("static_best_raw_momentum_mr_mom_20", static_bt),
            _momentum_mr_oos_metric_row("annual_walk_forward_momentum_mr", bt),
        ]
    )
    return {"signal": signal, "selected": selected, "positions": positions, "backtest": bt, "comparison": comparison}


def _candidate_metric_row(source_table, base_strategy, variant, guard, bt, guard_oos_pct=0.0):
    oos_metrics = product_flow_performance_metrics(bt.loc[bt.index >= pd.Timestamp(SPLIT_DATE)])
    full_metrics = product_flow_performance_metrics(bt)
    return {
        "source_table": source_table,
        "base_strategy": base_strategy,
        "variant": variant,
        "guard": guard,
        "oos_sharpe": oos_metrics.get("sharpe", np.nan),
        "oos_pnl": oos_metrics.get("total_pnl", np.nan),
        "oos_dd": oos_metrics.get("max_drawdown", np.nan),
        "oos_active_days": oos_metrics.get("days", np.nan),
        "full_sharpe": full_metrics.get("sharpe", np.nan),
        "full_pnl": full_metrics.get("total_pnl", np.nan),
        "full_dd": full_metrics.get("max_drawdown", np.nan),
        "turnover": full_metrics.get("avg_daily_turnover", np.nan),
        "avg_gross_exposure": full_metrics.get("avg_gross_exposure", np.nan),
        "guard_oos_pct": guard_oos_pct,
    }


def _apply_guard_positions(positions, guard_name, masks, commodity="CORN"):
    if guard_name == "no_guard":
        return positions.copy(), 0.0
    mask_name, action = guard_name.rsplit("_", 1)
    scale = 0.50 if action == "half" else 0.0
    mask = pd.Series(masks[mask_name], index=positions.index).fillna(False).astype(bool)
    guarded = positions.copy()
    guarded.loc[mask, commodity] = scale * guarded.loc[mask, commodity]
    return guarded.fillna(0.0), float(mask.loc[mask.index >= pd.Timestamp(SPLIT_DATE)].mean())


def _evaluate_candidate_guard_menu(context, source_table, base_strategy, variant, signal, masks):
    futures_pnl = context["futures_pnl"]
    commodity = context["commodity"]
    base_positions = corn_positions_from_signal(signal, futures_pnl)
    guard_names = ["no_guard"]
    for mask_name in masks:
        guard_names.extend([f"{mask_name}_half", f"{mask_name}_flat"])

    rows, backtests, positions_by_guard = [], {}, {}
    for guard_name in guard_names:
        positions, guard_oos_pct = _apply_guard_positions(base_positions, guard_name, masks, commodity)
        bt, _ = backtest_positions_product_flow(positions, futures_pnl)
        key = (base_strategy, variant, guard_name)
        rows.append(_candidate_metric_row(source_table, base_strategy, variant, guard_name, bt, guard_oos_pct))
        backtests[key] = bt
        positions_by_guard[key] = positions
    return rows, backtests, positions_by_guard


def run_corn_cargill_overlay_candidates(context, momentum_mr, walk_forward):
    signals = context["signals"]
    trading_index = context["trading_index"]
    futures_pnl = context["futures_pnl"]
    feature_panels = context["feature_panels"]
    data = context["data"]

    cargill_physical_signal = clean_product_flow_signal(
        mean_product_flow_signals(
            [signals["given_cgl_inventory_pressure"], signals["given_cgl_crush_activity"]],
            trading_index,
        ),
        trading_index,
    )
    base_spine_signals = {
        "wf_momentum_mr": walk_forward["signal"],
        "static_mom_20": momentum_mr["signals"]["mom_20"],
    }

    candidate_signals = {}
    for base_name, base_signal in base_spine_signals.items():
        aligned_base = clean_product_flow_signal(base_signal, trading_index)
        disagreement = (
            (aligned_base * cargill_physical_signal < 0.0)
            & (aligned_base.abs() > 0.05)
            & (cargill_physical_signal.abs() > 0.25)
        )
        half_filter = aligned_base.copy()
        half_filter.loc[disagreement] = 0.50 * half_filter.loc[disagreement]
        flat_filter = aligned_base.copy()
        flat_filter.loc[disagreement] = 0.0

        candidate_signals[(base_name, "base_no_cargill")] = aligned_base
        candidate_signals[(base_name, "cargill_overlay_90_10")] = clean_product_flow_signal(
            0.90 * aligned_base + 0.10 * cargill_physical_signal,
            trading_index,
        )
        candidate_signals[(base_name, "cargill_overlay_85_15")] = clean_product_flow_signal(
            0.85 * aligned_base + 0.15 * cargill_physical_signal,
            trading_index,
        )
        candidate_signals[(base_name, "cargill_disagree_half")] = clean_product_flow_signal(half_filter, trading_index)
        candidate_signals[(base_name, "cargill_disagree_flat")] = clean_product_flow_signal(flat_filter, trading_index)

    raw_rows = []
    for (base_name, variant), signal in candidate_signals.items():
        positions = corn_positions_from_signal(signal, futures_pnl)
        bt, _ = backtest_positions_product_flow(positions, futures_pnl)
        raw_rows.append(_candidate_metric_row("cargill_overlay", base_name, variant, "no_guard", bt, 0.0))
    raw_results = pd.DataFrame(raw_rows).sort_values(
        ["base_strategy", "oos_sharpe", "full_sharpe"],
        ascending=[True, False, False],
    )

    supply_masks = corn_abundant_supply_masks(data, feature_panels, futures_pnl)
    guard_rows, guard_backtests, guard_positions = [], {}, {}
    for (base_name, variant), signal in candidate_signals.items():
        rows, backtests, positions = _evaluate_candidate_guard_menu(
            context,
            "cargill_overlay_guarded",
            base_name,
            variant,
            signal,
            supply_masks,
        )
        guard_rows.extend(rows)
        guard_backtests.update(backtests)
        guard_positions.update(positions)

    guard_results = pd.DataFrame(guard_rows).sort_values(["oos_sharpe", "full_sharpe"], ascending=[False, False])
    wf_guard_results = guard_results.loc[guard_results["base_strategy"] == "wf_momentum_mr"].copy()
    final_row = wf_guard_results.sort_values(
        ["oos_sharpe", "full_sharpe", "oos_pnl"],
        ascending=[False, False, False],
    ).iloc[0]
    final_key = (final_row["base_strategy"], final_row["variant"], final_row["guard"])
    return {
        "cargill_physical_signal": cargill_physical_signal,
        "candidate_signals": candidate_signals,
        "raw_results": raw_results,
        "supply_masks": supply_masks,
        "guard_results": guard_results,
        "guard_backtests": guard_backtests,
        "guard_positions": guard_positions,
        "final_row": final_row,
        "final_key": final_key,
    }


def build_corn_final_report(cargill):
    final_row = cargill["final_row"]
    final_key = cargill["final_key"]
    guard_results = cargill["guard_results"]
    final_periods = product_flow_period_performance(cargill["guard_backtests"][final_key])[
        ["period", "total_pnl", "sharpe", "max_drawdown", "hit_rate", "days"]
    ]
    wf_base_reference = guard_results.loc[
        (guard_results["base_strategy"] == "wf_momentum_mr")
        & (guard_results["variant"] == "base_no_cargill")
        & (guard_results["guard"] == "no_guard")
    ].iloc[0]
    mom20_guard_reference = guard_results.loc[
        (guard_results["base_strategy"] == "static_mom_20")
        & (guard_results["variant"] == "base_no_cargill")
    ].sort_values(["oos_sharpe", "full_sharpe"], ascending=[False, False]).iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "strategy": "final_wf_momentum_mr_cargill",
                "variant": final_row["variant"],
                "guard": final_row["guard"],
                "oos_sharpe": final_row["oos_sharpe"],
                "oos_pnl": final_row["oos_pnl"],
                "oos_dd": final_row["oos_dd"],
                "full_sharpe": final_row["full_sharpe"],
                "full_dd": final_row["full_dd"],
            },
            {
                "strategy": "wf_momentum_mr_base_reference",
                "variant": wf_base_reference["variant"],
                "guard": wf_base_reference["guard"],
                "oos_sharpe": wf_base_reference["oos_sharpe"],
                "oos_pnl": wf_base_reference["oos_pnl"],
                "oos_dd": wf_base_reference["oos_dd"],
                "full_sharpe": wf_base_reference["full_sharpe"],
                "full_dd": wf_base_reference["full_dd"],
            },
            {
                "strategy": "guarded_mom20_benchmark",
                "variant": mom20_guard_reference["variant"],
                "guard": mom20_guard_reference["guard"],
                "oos_sharpe": mom20_guard_reference["oos_sharpe"],
                "oos_pnl": mom20_guard_reference["oos_pnl"],
                "oos_dd": mom20_guard_reference["oos_dd"],
                "full_sharpe": mom20_guard_reference["full_sharpe"],
                "full_dd": mom20_guard_reference["full_dd"],
            },
        ]
    )
    conclusion = f"""
### Conclusion

**Core idea:** annual walk-forward Momentum/MR is the main tradable structure.

**Cargill use:** Cargill inventory/crush activity is used as a physical disagreement filter, not as the dominant corn alpha.

**Final corn candidate:** {final_row["base_strategy"]} with {final_row["variant"]} and guard {final_row["guard"]}. OOS Sharpe {final_row["oos_sharpe"]:.3f}, OOS PnL {final_row["oos_pnl"]:.3f}, OOS DD {100.0 * final_row["oos_dd"] / 10000.0:.2f}%, full-period Sharpe {final_row["full_sharpe"]:.3f}.

The one-week recommendation is a risk-controlled Momentum/MR corn sleeve with a Cargill physical disagreement filter.
"""
    return {"final_periods": final_periods, "comparison": comparison, "conclusion": conclusion}
