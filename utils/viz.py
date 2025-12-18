from typing import Optional, Tuple
import math

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go



DEFAULT_TEMPLATE = "plotly_white"


def ensure_plotly():
    """Lazy import plotly so newly installed packages are detected without restarting.

    如果在启动 Streamlit 后才安装 plotly，此函数会在运行时再次尝试导入，避免必须重启进程。
    """
    global px, go
    if px is None or go is None:
        try:  # 尝试在运行时导入
            import importlib
            _px = importlib.import_module("plotly.express")
            _go = importlib.import_module("plotly.graph_objects")
            px = _px
            go = _go
        except Exception:
            raise RuntimeError("未安装 plotly，请先安装：pip install plotly")


def has_plotly() -> bool:
    try:
        ensure_plotly()
        return True
    except Exception:
        return False


def fig_units_energy(df_energy: pd.DataFrame, top_n: int = 15, template: str = DEFAULT_TEMPLATE):
    ensure_plotly()
    d = df_energy.copy()
    d = d.sort_values("total_kgco2e", ascending=False).head(top_n)
    fig = px.bar(
        d,
        x="total_kgco2e",
        y="unit_name",
        orientation="h",
        text="total_kgco2e",
        color="total_kgco2e",
        color_continuous_scale="Reds",
        template=template,
        title=f"装置公用工程 CO2e 总量（Top {top_n}）",
        labels={"total_kgco2e": "kgCO2e", "unit_name": "装置"},
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=80, r=40, t=60, b=40),
        font=dict(family="Microsoft YaHei, Arial", size=12),
    )
    return fig


def fig_pareto_units(df_units: pd.DataFrame, template: str = DEFAULT_TEMPLATE):
    ensure_plotly()
    d = df_units[["unit_name", "total_input_kgco2e"]].copy()
    d = d.sort_values("total_input_kgco2e", ascending=False)
    if d["total_input_kgco2e"].sum() <= 0:
        d["cum_share"] = 0.0
    else:
        d["cum_share"] = d["total_input_kgco2e"].cumsum() / d["total_input_kgco2e"].sum()

    fig = go.Figure()
    fig.add_bar(x=d["unit_name"], y=d["total_input_kgco2e"], name="投入 CO2e (kg)")
    fig.add_scatter(
        x=d["unit_name"],
        y=d["cum_share"] * 100.0,
        mode="lines+markers",
        name="累计占比 %",
        yaxis="y2",
    )
    fig.update_layout(
        template=template,
        title="Pareto：装置投入 CO2e 累计占比",
        xaxis=dict(tickangle=30),
        yaxis=dict(title="kgCO2e"),
        yaxis2=dict(title="%", overlaying="y", side="right", range=[0, 100]),
        margin=dict(l=60, r=60, t=60, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Microsoft YaHei, Arial", size=12),
    )
    return fig


def fig_streams_top(df_stream: Optional[pd.DataFrame], top_n: int = 10, template: str = DEFAULT_TEMPLATE):
    ensure_plotly()
    if df_stream is None or df_stream.empty:
        return go.Figure()
    d = df_stream.sort_values("factor_stream", ascending=False).head(top_n).copy()
    fig = px.bar(
        d,
        x="factor_stream",
        y="stream_id",
        orientation="h",
        text="factor_stream",
        color="factor_stream",
        color_continuous_scale="Blues",
        template=template,
        title=f"碳足迹因子 Top {top_n}",
        labels={"factor_stream": "kgCO2e/单位", "stream_id": "流"},
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=120, r=40, t=60, b=40),
        font=dict(family="Microsoft YaHei, Arial", size=12),
    )
    return fig


def fig_intensity_scatter(df_units: pd.DataFrame, template: str = DEFAULT_TEMPLATE):
    ensure_plotly()
    d = df_units.copy()
    d["intensity"] = d.apply(
        lambda r: (float(r["total_input_kgco2e"]) / float(r["total_output_qty"])) if float(r["total_output_qty"]) > 0 else math.nan,
        axis=1,
    )
    d = d.dropna(subset=["intensity"])  # 移除无产量或无强度的点
    fig = px.scatter(
        d,
        x="total_output_qty",
        y="intensity",
        size="total_input_kgco2e",
        hover_data=["unit_name", "total_input_kgco2e"],
        template=template,
        title="强度散点：产量 vs 排放强度",
        labels={"total_output_qty": "总产量", "intensity": "kgCO2e/产量单位"},
        color="intensity",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(margin=dict(l=60, r=40, t=60, b=60), font=dict(family="Microsoft YaHei, Arial", size=12))
    return fig


def fig_intensity_box(df_units: pd.DataFrame, template: str = DEFAULT_TEMPLATE):
    ensure_plotly()
    d = df_units.copy()
    d["intensity"] = d.apply(
        lambda r: (float(r["total_input_kgco2e"]) / float(r["total_output_qty"])) if float(r["total_output_qty"]) > 0 else math.nan,
        axis=1,
    )
    d = d.dropna(subset=["intensity"])  # 去掉无意义的点
    fig = px.box(
        d,
        y="intensity",
        points="outliers",
        template=template,
        title="排放强度分布（箱线图）",
        labels={"intensity": "kgCO2e/产量单位"},
        color_discrete_sequence=["#5B8FF9"],
    )
    fig.update_layout(margin=dict(l=60, r=40, t=60, b=40), font=dict(family="Microsoft YaHei, Arial", size=12))
    return fig


def fig_intensity_scatter_highlight(df_units: pd.DataFrame, template: str = DEFAULT_TEMPLATE) -> Tuple[object, pd.DataFrame]:
    """
    强度散点图（同 fig_intensity_scatter），并对 IQR 离群点进行红色高亮。
    返回 (Figure, 离群点DataFrame)。
    """
    ensure_plotly()
    d = df_units.copy()
    d["intensity"] = d.apply(
        lambda r: (float(r["total_input_kgco2e"]) / float(r["total_output_qty"])) if float(r["total_output_qty"]) > 0 else math.nan,
        axis=1,
    )
    base = d.dropna(subset=["intensity"]).copy()
    fig = px.scatter(
        base,
        x="total_output_qty",
        y="intensity",
        size="total_input_kgco2e",
        hover_data=["unit_name", "total_input_kgco2e"],
        template=template,
        title="强度散点：产量 vs 排放强度（含异常高亮）",
        labels={"total_output_qty": "总产量", "intensity": "kgCO2e/产量单位"},
        color="intensity",
        color_continuous_scale="Viridis",
    )
    # IQR 离群
    q1 = base["intensity"].quantile(0.25)
    q3 = base["intensity"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    outs = base[(base["intensity"] > upper) | (base["intensity"] < lower)].copy()
    if not outs.empty:
        fig.add_trace(
            go.Scatter(
                x=outs["total_output_qty"],
                y=outs["intensity"],
                mode="markers+text",
                name="异常点",
                marker=dict(color="red", size=10, line=dict(color="white", width=0.8)),
                text=outs["unit_name"],
                textposition="top center",
                hovertemplate="装置=%{text}<br>产量=%{x}<br>强度=%{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(margin=dict(l=60, r=40, t=60, b=60), font=dict(family="Microsoft YaHei, Arial", size=12))
    return fig, outs


def to_png_bytes(fig) -> Optional[bytes]:
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        # 需要 kaleido 才能导出图像
        return None


def _rgba(color_rgb: str, alpha: float) -> str:
    if color_rgb.startswith("rgb(") and color_rgb.endswith(")"):
        return color_rgb.replace("rgb(", "rgba(").replace(")", f", {alpha})")
    return color_rgb


def fig_sankey_transfers(model, df_stream: Optional[pd.DataFrame], template: str = DEFAULT_TEMPLATE, use_co2e: bool = True):
    """
    基于 routing 构建装置间流向桑基图。
    - 若提供 df_stream（包含 stream_id=material@unit 和 factor_stream），则可计算每条流的 CO2e。
    - use_co2e=True 时，按 CO2e 作为 link value（否则按 amount）。
    """
    ensure_plotly()

    # 构造 stream 因子索引： (material_id, source_unit_id) -> factor
    factor_by_stream = {}
    if df_stream is not None and not df_stream.empty and "stream_id" in df_stream.columns:
        for _, r in df_stream.iterrows():
            sid = str(r["stream_id"])  # 形如 M_X@U1
            if "@" in sid:
                mid, suid = sid.split("@", 1)
                try:
                    factor_by_stream[(mid, suid)] = float(r["factor_stream"])  # kgCO2e / unit
                except Exception:
                    continue

    # 节点：所有装置
    units = list(getattr(model, "units").keys())
    label_by_id = {uid: getattr(model, "units")[uid].unit_name for uid in units}
    node_index = {uid: i for i, uid in enumerate(units)}

    # 边：根据 routing
    links_src = []
    links_tgt = []
    links_val = []
    links_lab = []

    routing = getattr(model, "routing", {}) or {}
    for (consumer_uid, material_id), sources in routing.items():
        for source_uid, amount in sources:
            try:
                amt = float(amount)
            except Exception:
                continue
            if amt <= 0:
                continue
            value = amt
            label = f"{material_id}: {amt:g}"
            if use_co2e and (material_id, source_uid) in factor_by_stream:
                f = factor_by_stream[(material_id, source_uid)]
                co2e = amt * f
                value = max(co2e, 0.0)
                label = f"{material_id}: {co2e:,.0f} kgCO2e"
            if source_uid in node_index and consumer_uid in node_index:
                links_src.append(node_index[source_uid])
                links_tgt.append(node_index[consumer_uid])
                links_val.append(value)
                links_lab.append(label)

    if not links_src:
        return go.Figure()

    # 节点颜色：按节点强度（入+出）映射到蓝色系
    node_strength = {uid: 0.0 for uid in units}
    for s, t, v in zip(links_src, links_tgt, links_val):
        node_strength[units[s]] += float(v)
        node_strength[units[t]] += float(v)
    ns_vals = list(node_strength.values())
    ns_min, ns_max = (min(ns_vals), max(ns_vals)) if ns_vals else (0.0, 1.0)
    node_colors = []
    for uid in units:
        if ns_max > ns_min:
            ratio = (node_strength[uid] - ns_min) / (ns_max - ns_min)
        else:
            ratio = 0.0
        color = px.colors.sample_colorscale("Blues", [ratio])[0]
        node_colors.append(_rgba(color, 0.85))

    # 链接颜色：按值映射到红色系
    lv_min, lv_max = (min(links_val), max(links_val)) if links_val else (0.0, 1.0)
    link_colors = []
    for v in links_val:
        if lv_max > lv_min:
            r = (float(v) - lv_min) / (lv_max - lv_min)
        else:
            r = 0.0
        c = px.colors.sample_colorscale("Reds", [r])[0]
        link_colors.append(_rgba(c, 0.7))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=15,
                    thickness=16,
                    line=dict(color="rgba(0,0,0,0.25)", width=0.5),
                    label=[label_by_id[u] for u in units],
                    color=node_colors,
                ),
                link=dict(
                    source=links_src,
                    target=links_tgt,
                    value=links_val,
                    label=links_lab,
                    color=link_colors,
                ),
            )
        ]
    )
    fig.update_layout(
        template=template,
        title="装置间流向桑基图（颜色按规模映射：节点=蓝色强度，链接=红色强度）",
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Microsoft YaHei, Arial", size=12),
    )
    return fig
