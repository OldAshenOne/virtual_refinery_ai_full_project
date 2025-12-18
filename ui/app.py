import os
import sys
import tempfile
from io import BytesIO
import json

import pandas as pd
import streamlit as st

# 让 project 包可导入
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from services.excel_loader import load_company_from_excel
from services.solver_linear import solve_linear
from services.solver_iterative import solve_iterative
from services.solver_stream import solve_stream_factors
from services.reporting import build_reports
from services.energy_report import build_energy_summary
from services.ai_report import generate_narrative
from services.llm_adapter import chat_with_llm
from utils import viz
from config import load_llm_config, GRANULARITY, SCOPES, MYSQL

try:
    from db.mysql_store import MySQLStore
    _db_error = None
except Exception as _e:
    MySQLStore = None  # type: ignore
    _db_error = _e


st.set_page_config(page_title="智能炼厂碳足迹数据核算及分析小程序", layout="wide")
st.title("智能炼厂碳足迹数据核算及分析小程序")
st.caption("Excel 导入 · 列映射与校验 · 计算 · 可视化 · 中文 AI 总结")


# ---------- 辅助方法 ----------
@st.cache_data(show_spinner=False)
def _persist_upload_to_tmp(data: bytes, suffix: str = ".xlsx") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


@st.cache_data(show_spinner=True)
def _run_pipeline(tmp_path: str):
    model = load_company_from_excel(tmp_path)
    f_lin = solve_linear(model)
    f_it = solve_iterative(model)
    f_stream = solve_stream_factors(model)
    df_lin, df_it, df_stream, df_units = build_reports(model, f_lin, f_it, f_stream)
    df_energy = build_energy_summary(model)
    return {
        "model": model,
        "df_lin": df_lin,
        "df_it": df_it,
        "df_stream": df_stream,
        "df_units": df_units,
        "df_energy": df_energy,
    }


def _sheet_specs():
    return {
        "materials": {"required": ["material_id", "material_name", "unit", "category"]},
        "fixed_factors": {"required": ["material_id", "factor_per_unit"]},
        "units": {"required": ["unit_id", "unit_name"]},
        "consumption": {"required": ["unit_id", "material_id", "amount"]},
        "production": {"required": ["unit_id", "material_id", "amount"]},
        "carryover": {"required": ["material_id", "factor_init"]},
        "routing": {"required": ["consumer_unit_id", "material_id", "source_unit_id", "amount", "unit"]},
    }


def _guess_map(sheet: str, df: pd.DataFrame) -> dict:
    synonyms = {
        "material_id": ["material_id", "物料id", "物料编码", "物料编号", "mid"],
        "material_name": ["material_name", "物料名称", "名称", "mname"],
        "unit": ["unit", "单位", "uom"],
        "category": ["category", "类别", "分类"],
        "factor_per_unit": ["factor_per_unit", "排放因子", "单耗因子", "因子", "kgco2e/单位"],
        "unit_id": ["unit_id", "装置id", "装置编码", "uid"],
        "unit_name": ["unit_name", "装置名称", "单元名称", "uname"],
        "amount": ["amount", "数量", "用量", "产量", "消耗量", "量"],
        "factor_init": ["factor_init", "初始因子", "初始系数", "carry_init"],
        "consumer_unit_id": ["consumer_unit_id", "消费装置", "去向装置", "目标装置", "to_unit", "to"],
        "source_unit_id": ["source_unit_id", "来源装置", "供给装置", "from_unit", "from"],
    }
    mapping = {}
    cols = [str(c).strip() for c in df.columns]
    lowmap = {c.lower().replace(" ", ""): c for c in cols}
    for need in _sheet_specs()[sheet]["required"]:
        candidates = synonyms.get(need, [need])
        chosen = None
        for cand in candidates:
            key = str(cand).lower().replace(" ", "")
            if key in lowmap:
                chosen = lowmap[key]
                break
        mapping[need] = chosen
    return mapping


def _validate_standardized(std: dict) -> list:
    issues = []
    specs = _sheet_specs()
    for sheet in ["materials", "fixed_factors", "units", "consumption", "production", "carryover"]:
        if sheet not in std or std[sheet] is None:
            issues.append(f"缺少必要 sheet：{sheet}")
    if issues:
        return issues
    for sheet, meta in specs.items():
        if sheet not in std or std[sheet] is None:
            continue
        df = std[sheet]
        for col in meta["required"]:
            if col not in df.columns:
                issues.append(f"{sheet} 缺少必需列：{col}")
        if df.shape[0] == 0 and sheet in ["materials", "units"]:
            issues.append(f"{sheet} 为空")
    for sheet in ["consumption", "production"]:
        df = std.get(sheet)
        if df is None:
            continue
        if "amount" in df.columns:
            bad = df[pd.to_numeric(df["amount"], errors="coerce").isna()]
            if not bad.empty:
                issues.append(f"{sheet}.amount 存在非数字或空值 {len(bad)} 行")
            neg = df[pd.to_numeric(df["amount"], errors="coerce") < 0]
            if not neg.empty:
                issues.append(f"{sheet}.amount 存在负值 {len(neg)} 行（不支持负流）")
    df = std.get("carryover")
    if df is not None and "factor_init" in df.columns:
        bad = df[pd.to_numeric(df["factor_init"], errors="coerce").isna()]
        if not bad.empty:
            issues.append(f"carryover.factor_init 存在非数字或空值 {len(bad)} 行")
        neg = df[pd.to_numeric(df["factor_init"], errors="coerce") < 0]
        if not neg.empty:
            issues.append(f"carryover.factor_init 存在负值 {len(neg)} 行")
    mat_ids = set(std["materials"]["material_id"].astype(str)) if std.get("materials") is not None else set()
    unit_ids = set(std["units"]["unit_id"].astype(str)) if std.get("units") is not None else set()
    for sheet in ["consumption", "production"]:
        df = std.get(sheet)
        if df is None:
            continue
        if (~df["material_id"].astype(str).isin(mat_ids)).any():
            issues.append(f"{sheet} 中存在未知 material_id")
        if (~df["unit_id"].astype(str).isin(unit_ids)).any():
            issues.append(f"{sheet} 中存在未知 unit_id")
    return issues


def _build_standardized_excel(raw_sheets: dict, mappings: dict) -> tuple[str, dict]:
    std = {}
    specs = _sheet_specs()
    for sheet, meta in specs.items():
        if sheet not in raw_sheets or raw_sheets[sheet] is None:
            std[sheet] = None
            continue
        df = raw_sheets[sheet].copy()
        rename_map = {}
        for need in meta["required"]:
            chosen = mappings.get(sheet, {}).get(need)
            if chosen:
                rename_map[chosen] = need
        df = df.rename(columns=rename_map)
        keep = [c for c in meta["required"] if c in df.columns]
        df = df[keep]
        std[sheet] = df
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    tmp_path = tmp.name
    tmp.close()
    with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
        for sheet, df in std.items():
            if df is None:
                continue
            df.to_excel(writer, sheet_name=sheet, index=False)
    return tmp_path, std


# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("配置与帮助")
    cfg = load_llm_config()
    st.caption("LLM 提供方与模型（仅展示）")
    st.code(f"Provider: {cfg.provider}\nBase URL: {cfg.base_url}\nModel: {cfg.model}", language="bash")

    st.markdown("### LLM 超参数")
    llm_params = st.session_state.get("llm_params", {"temperature": cfg.temperature, "top_p": 0.9, "top_k": 20, "max_tokens": None})
    llm_params["temperature"] = st.slider("temperature", 0.0, 1.5, float(llm_params.get("temperature", 0.2)), 0.05)
    llm_params["top_p"] = st.slider("top_p", 0.1, 1.0, float(llm_params.get("top_p", 0.9)), 0.05)
    llm_params["top_k"] = st.number_input("top_k (可选)", min_value=1, max_value=50, value=int(llm_params.get("top_k", 20)))
    llm_params["max_tokens"] = llm_params.get("max_tokens")
    st.session_state["llm_params"] = llm_params

    if st.button("测试 LLM 连接"):
        try:
            with st.spinner("正在测试 LLM 连接…"):
                _ = generate_narrative(pd.DataFrame({"unit_name": ["PING"], "total_input_kgco2e": [1.0], "total_output_qty": [1.0]}), None, llm_params)
            st.success("LLM 连接正常")
        except Exception as e:
            st.error(f"LLM 连接失败：{e}")

    st.markdown("---")
    st.subheader("风格设置")
    template = st.selectbox("图表主题", options=["plotly_white", "seaborn", "simple_white", "ggplot2", "plotly_dark"], index=0)
    st.session_state["plotly_template"] = template
    if not viz.has_plotly():
        st.warning("未检测到 plotly，将使用简化图表展示。安装：pip install plotly kaleido")

    st.markdown("---")
    st.subheader("MySQL 状态")
    if _db_error is not None:
        st.error("无法使用 MySQL：请安装 pymysql 并在 config.py 设置连接信息。")
        db_store = None
    else:
        try:
            db_store = MySQLStore()
            st.caption(f"连接：{MYSQL.user}@{MYSQL.host}:{MYSQL.port}/{MYSQL.database}")
            if st.button("初始化数据库（建库建表）"):
                with st.spinner("正在创建数据库与表…"):
                    db_store.init_schema()
                st.success("初始化完成")
        except Exception as e:
            db_store = None
            st.error(f"无法初始化 MySQL 连接：{e}")

    st.markdown("---")
    st.caption(f"当前设置：粒度={GRANULARITY}，Scope={','.join(SCOPES)}")
    sample_path = os.path.join(PROJECT_DIR, "..", "virtual_refinery_data.xlsx")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            st.download_button(label="下载示例模板 Excel", data=f.read(), file_name="virtual_refinery_data.xlsx")


# ---------- 主体标签页 ----------
tab_upload, tab_map, tab_calc, tab_factors, tab_viz, tab_ai, tab_dl, tab_hist = st.tabs([
    "上传与预览", "列映射与校验", "计算与结果", "因子与物料", "可视化", "AI 总结", "下载", "历史记录",
])


with tab_upload:
    uploaded = st.file_uploader("上传数据文件（Excel，包含 materials/fixed_factors/units/consumption/production/carryover/routing）", type=["xlsx"])
    if uploaded is not None:
        st.session_state["uploaded_name"] = getattr(uploaded, "name", "uploaded.xlsx")
        st.session_state["raw_tmp_path"] = _persist_upload_to_tmp(uploaded.getvalue())
        try:
            xls = pd.ExcelFile(uploaded)
            st.success("已读取 Excel 文件。Sheet 列表：")
            st.write(xls.sheet_names)
            st.session_state["raw_sheets"] = {s: pd.read_excel(xls, s) for s in xls.sheet_names}
            for sheet in ["units", "consumption", "production", "materials", "fixed_factors", "carryover", "routing"]:
                if sheet in xls.sheet_names:
                    st.subheader(f"预览：{sheet}")
                    st.dataframe(pd.read_excel(xls, sheet).head())
        except Exception as e:
            st.error(f"读取 Excel 失败：{e}")
    else:
        st.info("请上传 Excel 文件，或在侧边栏下载示例模板。")


with tab_map:
    raw_sheets = st.session_state.get("raw_sheets")
    if not raw_sheets:
        st.info("请先在“上传与预览”页上传 Excel 文件。")
    else:
        st.write("为每个 Sheet 选择/确认列映射到标准字段（可勾选自动猜测）。")
        specs = _sheet_specs()
        mappings = st.session_state.get("mappings", {})
        auto = st.checkbox("自动猜测映射", value=True)
        for sheet, meta in specs.items():
            st.markdown(f"#### {sheet}")
            df = raw_sheets.get(sheet)
            if df is None:
                st.caption("（该 sheet 不存在，允许为空，例如 routing）")
                continue
            if sheet not in mappings:
                mappings[sheet] = {}
            if auto:
                guessed = _guess_map(sheet, df)
                for k, v in guessed.items():
                    mappings[sheet].setdefault(k, v)
            cols = [str(c) for c in df.columns]
            c1, c2, c3, c4 = st.columns(4)
            cols_layout = [c1, c2, c3, c4]
            for i, need in enumerate(meta["required"]):
                with cols_layout[i % 4]:
                    sel = st.selectbox(
                        f"{sheet}.{need}",
                        options=["<不映射>"] + cols,
                        index=(cols.index(mappings[sheet][need]) + 1) if mappings[sheet].get(need) in cols else 0,
                        key=f"map_{sheet}_{need}",
                    )
                    mappings[sheet][need] = None if sel == "<不映射>" else sel
        st.session_state["mappings"] = mappings

        if st.button("标准化并校验"):
            with st.spinner("正在标准化并校验…"):
                tmp_path, std = _build_standardized_excel(raw_sheets, mappings)
                issues = _validate_standardized(std)
                st.session_state["std_excel_path"] = tmp_path
                st.session_state["std_std"] = std
                st.session_state["validation_issues"] = issues
            if issues:
                st.error("校验未通过：")
                for it in issues:
                    st.write("- ", it)
            else:
                st.success("校验通过，可前往“计算与结果”执行计算。")


with tab_calc:
    run_btn = st.button("开始计算/重新计算")
    if run_btn:
        with st.spinner("正在计算与生成报表…"):
            use_path = st.session_state.get("std_excel_path") or st.session_state.get("raw_tmp_path")
            if not use_path:
                st.error("未找到可用数据，请先上传或完成标准化。")
            else:
                results = _run_pipeline(use_path)
                st.session_state["results"] = results
                st.success("计算完成")

    results = st.session_state.get("results")
    if results:
        df_units = results["df_units"]
        df_stream = results["df_stream"]
        df_energy = results["df_energy"]

        c1, c2, c3 = st.columns(3)
        with c1:
            total_input = float(df_units["total_input_kgco2e"].sum()) if not df_units.empty else 0.0
            st.metric("总投入 CO2e (kg)", f"{total_input:,.0f}")
        with c2:
            n_units = int(df_units.shape[0])
            st.metric("装置数", str(n_units))
        with c3:
            n_streams = int(0 if df_stream is None else df_stream.shape[0])
            st.metric("流（流因子）数", str(n_streams))

        st.subheader("装置汇总（单位级）")
        st.dataframe(df_units)
        if db_store is not None:
            if st.button("保存到 MySQL"):
                try:
                    use_path = st.session_state.get("std_excel_path") or st.session_state.get("raw_tmp_path")
                    file_name = st.session_state.get("uploaded_name", "uploaded.xlsx")
                    from db.mysql_store import MySQLStore as _MS
                    fhash = _MS.file_sha256(use_path) if use_path else ""
                    params = {
                        "granularity": GRANULARITY,
                        "scopes": list(SCOPES),
                        "llm_provider": cfg.provider,
                        "llm_model": cfg.model,
                        "source_file": file_name,
                    }
                    run_id = db_store.create_run(file_name, fhash, params, status="completed")
                    db_store.save_units_results(run_id, df_units)
                    db_store.save_stream_results(run_id, results.get("df_stream"))
                    narrative = st.session_state.get("narrative")
                    if narrative:
                        db_store.save_ai(run_id, cfg.model, narrative)
                    st.success(f"已保存到 MySQL，run_id={run_id}")
                except Exception as e:
                    st.error(f"保存失败：{e}")
    else:
        st.info("请点击上方按钮计算。")


with tab_factors:
    results = st.session_state.get("results")
    if not results:
        st.info("请先在“计算与结果”页完成计算。")
    else:
        df_lin = results.get("df_lin")
        df_it = results.get("df_it")
        model = results.get("model")
        mats = pd.DataFrame(
            [
                dict(material_id=m.material_id, material_name=m.name, unit=m.unit, category=m.category)
                for m in model.materials.values()
            ]
        ) if getattr(model, "materials", None) else pd.DataFrame(columns=["material_id", "material_name", "unit", "category"])
        df = pd.merge(df_lin, df_it, on="material_id", how="outer") if df_lin is not None and df_it is not None else (df_lin or df_it)
        df = pd.merge(mats, df, on="material_id", how="left") if not mats.empty else df
        st.subheader("物料碳足迹因子（kgCO2e/单位）")
        if df is None or df.empty:
            st.info("暂无因子结果。")
        else:
            sort_col = "factor_linear" if "factor_linear" in df.columns else ("factor_iter" if "factor_iter" in df.columns else None)
            if sort_col:
                df = df.sort_values(sort_col, ascending=False)
            st.dataframe(df)
            with st.expander("结果说明"):
                st.write(
                    "- 因子含义：每单位物料所携带的碳足迹（kgCO2e/单位）。\n"
                    "- 线性解与迭代解通常应接近；若差异较大，提示循环/路由或初值设定需检查。\n"
                    "- 因子用于结转计算与路径分析，也可直接用于产品碳足迹核算。"
                )
            xbuf2 = BytesIO()
            with pd.ExcelWriter(xbuf2, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="material_factors", index=False)
            st.download_button("下载因子表（Excel）", data=xbuf2.getvalue(), file_name="material_factors.xlsx")


with tab_viz:
    results = st.session_state.get("results")
    if not results:
        st.info("请先在“计算与结果”页完成计算。")
    else:
        df_units = results["df_units"].copy()
        df_energy = results["df_energy"].copy()
        tmpl = st.session_state.get("plotly_template", "plotly_white")

        st.subheader("公用工程 CO2e 总量（Top N）")
        topn = st.slider("Top N", min_value=5, max_value=30, value=15, step=1)
        if not df_energy.empty and viz.has_plotly():
            fig1 = viz.fig_units_energy(df_energy, top_n=topn, template=tmpl)
            st.plotly_chart(fig1, use_container_width=True)
        elif not df_energy.empty:
            st.bar_chart(df_energy.sort_values("total_kgco2e", ascending=False).head(topn).set_index("unit_name")["total_kgco2e"])

        st.subheader("Pareto（碳排放贡献 80/20）")
        if not df_units.empty and viz.has_plotly():
            fig2 = viz.fig_pareto_units(df_units, template=tmpl)
            st.plotly_chart(fig2, use_container_width=True)

        if results.get("df_stream") is not None and not results["df_stream"].empty and viz.has_plotly():
            st.subheader("碳足迹因子 Top N")
            topn_s = st.slider("Top N", min_value=5, max_value=30, value=10, step=1)
            fig3 = viz.fig_streams_top(results["df_stream"], top_n=topn_s, template=tmpl)
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("装置间流向桑基图")
        if viz.has_plotly():
            fig_sankey = viz.fig_sankey_transfers(results["model"], results.get("df_stream"), template=tmpl, use_co2e=True)
            if len(fig_sankey.data) == 0:
                st.info("没有可用的 routing 数据，无法绘制桑基图。")
            else:
                st.plotly_chart(fig_sankey, use_container_width=True)
        else:
            st.info("未安装 plotly，无法绘制桑基图。")

        st.subheader("排放强度：散点与分布")
        if not df_units.empty and viz.has_plotly():
            highlight = st.checkbox("高亮强度异常点（IQR）", value=True)
            if highlight:
                fig4, _ = viz.fig_intensity_scatter_highlight(df_units, template=tmpl)
            else:
                fig4 = viz.fig_intensity_scatter(df_units, template=tmpl)
            st.plotly_chart(fig4, use_container_width=True)
            show_box = st.checkbox("显示排放强度箱线图（用于异常识别）", value=False)
            if show_box:
                fig5 = viz.fig_intensity_box(df_units, template=tmpl)
                st.plotly_chart(fig5, use_container_width=True)


with tab_ai:
    results = st.session_state.get("results")
    if not results:
        st.info("请先在“计算与结果”页完成计算。")
    else:
        df_units = results["df_units"]
        df_stream = results["df_stream"]
        params = st.session_state.get("llm_params", {})
        regen = st.button("生成/重新生成总结")
        if regen:
            try:
                with st.spinner("正在调用 DeepSeek‑V3 生成总结，请稍候…"):
                    st.session_state["narrative"] = generate_narrative(df_units, df_stream, llm_params=params)
            except Exception:
                st.error("AI 服务不可用或鉴权失败，请稍后再试。")
                st.session_state["narrative"] = ""
        st.text_area("AI 总结（DeepSeek‑V3）", st.session_state.get("narrative", ""), height=260)

        if st.session_state.get("narrative"):
            st.markdown("---")
            st.subheader("chat with agent")
            if "chat" not in st.session_state:
                st.session_state["chat"] = []

            user_q = st.chat_input("向 agent 追问…")
            if user_q:
                # 先插入用户消息
                st.session_state["chat"].append({"role": "user", "content": user_q.strip()})

                # 结构化摘要
                dfu = df_units.copy()
                dfu["total_input_kgco2e"] = pd.to_numeric(dfu["total_input_kgco2e"], errors="coerce").fillna(0.0)
                dfu["total_output_qty"] = pd.to_numeric(dfu["total_output_qty"], errors="coerce").fillna(0.0)
                total_units = int(dfu.shape[0])
                total_input = float(dfu["total_input_kgco2e"].sum())
                top_units = (
                    dfu.sort_values("total_input_kgco2e", ascending=False)
                    .head(5)[["unit_name", "total_input_kgco2e", "total_output_qty"]]
                    .to_dict(orient="records")
                )
                top_streams = []
                if df_stream is not None and not df_stream.empty:
                    dfs = df_stream.copy()
                    dfs["factor_stream"] = pd.to_numeric(dfs["factor_stream"], errors="coerce").fillna(0.0)
                    top_streams = (
                        dfs.sort_values("factor_stream", ascending=False)
                        .head(5)[["stream_id", "factor_stream"]]
                        .to_dict(orient="records")
                    )
                tmp = dfu.copy()
                tmp["intensity"] = tmp.apply(
                    lambda r: (float(r["total_input_kgco2e"]) / float(r["total_output_qty"])) if float(r["total_output_qty"]) > 0 else None,
                    axis=1,
                )
                tmp = tmp.dropna(subset=["intensity"])
                outliers = []
                if not tmp.empty:
                    q1 = tmp["intensity"].quantile(0.25)
                    q3 = tmp["intensity"].quantile(0.75)
                    iqr = q3 - q1
                    upper = q3 + 1.5 * iqr
                    lower = q1 - 1.5 * iqr
                    outs = tmp[(tmp["intensity"] > upper) | (tmp["intensity"] < lower)]
                    outliers = outs.sort_values("intensity", ascending=False)[["unit_name", "intensity"]].to_dict(orient="records")

                ctx = {
                    "total_units": total_units,
                    "total_input_kgco2e": total_input,
                    "top_units": top_units,
                    "top_streams": top_streams,
                    "intensity_outliers": outliers,
                }
                messages = [
                    {"role": "system", "content": "你是一名资深碳盘查与能效优化顾问，用简洁中文解答。"},
                    {"role": "user", "content": "以下为结构化摘要(JSON)，请据此回答后续问题：\n" + json.dumps(ctx, ensure_ascii=False)},
                    {"role": "user", "content": "以下是已生成的年度报告摘要，请据此回答后续问题：\n" + st.session_state["narrative"]},
                ]
                # 把历史对话全部串进去（含最新用户消息）
                for mm in st.session_state["chat"]:
                    messages.append({"role": mm["role"], "content": mm["content"]})

                # 调用 LLM
                try:
                    with st.spinner("AI 正在思考…"):
                        ans = chat_with_llm(messages, **params)
                    st.session_state["chat"].append({"role": "assistant", "content": ans})
                except Exception as e:
                    st.error(f"AI 服务不可用或鉴权失败，请稍后再试：{e}")
                    # 出错时移除刚刚插入的用户消息，避免下次重复发送
                    if st.session_state["chat"] and st.session_state["chat"][-1]["role"] == "user":
                        st.session_state["chat"].pop()
            for m in st.session_state["chat"]:
                with st.chat_message("user" if m["role"] == "user" else "assistant"):
                    st.markdown(m["content"])


with tab_dl:
    results = st.session_state.get("results")
    if not results:
        st.info("请先在“计算与结果”页完成计算。")
    else:
        xbuf = BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
            results["df_lin"].to_excel(writer, sheet_name="f_linear", index=False)
            results["df_it"].to_excel(writer, sheet_name="f_iterative", index=False)
            if results["df_stream"] is not None:
                results["df_stream"].to_excel(writer, sheet_name="f_stream", index=False)
            results["df_units"].to_excel(writer, sheet_name="units", index=False)
            results["df_energy"].to_excel(writer, sheet_name="energy", index=False)
        st.download_button(label="下载计算结果 Excel", data=xbuf.getvalue(), file_name="virtual_refinery_results_ui.xlsx")


with tab_hist:
    if MySQLStore is None:
        st.info("未安装 MySQL 依赖或初始化失败，无法读取历史记录。请安装 pymysql 并在侧边栏初始化数据库。")
    else:
        if db_store is None:
            st.info("请先在侧边栏完成数据库初始化。")
        else:
            try:
                runs = db_store.list_runs(limit=50)
                if not runs:
                    st.info("暂无历史记录。")
                else:
                    df_runs = pd.DataFrame(runs)
                    st.dataframe(df_runs)
                    sel = st.selectbox("选择 run_id 查看详情", options=[r["id"] for r in runs])
                    if st.button("加载详情"):
                        try:
                            dfu = db_store.load_units(int(sel))
                            st.subheader("单位级结果（历史）")
                            st.dataframe(dfu)
                            aistr = db_store.load_ai(int(sel))
                            if aistr:
                                st.subheader("AI 总结（历史）")
                                st.text_area("AI 总结（只读）", aistr, height=160)
                        except Exception as e:
                            st.error(f"加载失败：{e}")
            except Exception as e:
                st.error(f"读取历史记录失败：{e}")
