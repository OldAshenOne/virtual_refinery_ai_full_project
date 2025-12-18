from typing import Optional
import pandas as pd
from .llm_adapter import generate_with_llm


def _build_context(df_units: pd.DataFrame, df_stream: Optional[pd.DataFrame]) -> dict:
    ctx = {}
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
    ctx.update(dict(total_units=total_units, total_input_kgco2e=total_input, top_units=top_units))
    if df_stream is not None and not df_stream.empty:
        dfs = df_stream.copy()
        dfs["factor_stream"] = pd.to_numeric(dfs["factor_stream"], errors="coerce").fillna(0.0)
        top_streams = (
            dfs.sort_values("factor_stream", ascending=False)
            .head(5)[["stream_id", "factor_stream"]]
            .to_dict(orient="records")
        )
        ctx["top_streams"] = top_streams
    return ctx


def _prompt_zh(ctx: dict) -> str:
    top_units_txt = "\n".join(
        f"- {i+1}. {u['unit_name']}：投入 {u['total_input_kgco2e']:.0f} kgCO2e，产量 {u['total_output_qty']:.0f}"
        for i, u in enumerate(ctx.get("top_units", []))
    ) or "(无)"
    top_streams_txt = "\n".join(
        f"- {i+1}. {s['stream_id']}：因子 {s['factor_stream']:.4f} kgCO2e/单位"
        for i, s in enumerate(ctx.get("top_streams", []))
    ) or "(无)"

    return f"""
你是一名资深碳足迹核算与能效优化顾问，请基于以下数据背景，使用规范、简洁、通顺的中文，输出结构化分析与可执行建议。

【数据背景】
- 装置数：{ctx.get('total_units', 0)}
- 总投入CO2e：{ctx.get('total_input_kgco2e', 0):,.0f} kgCO2e
- Top装置（按年度投入CO2e）：
{top_units_txt}
- Top流（按单位因子）（如为空表示未统计流）：
{top_streams_txt}

【写作要求】
1. 用中文分条输出，避免空洞描述、避免夸张。
2. 先给出“总体结论（3-5条）”，再给出“关键发现（能耗/强度/集中度/异常）”。
3. 提供“优化建议（≥5条，有优先级：高/中/低）”，每条建议包含：方向（工艺/能源/管理/数据）、措施、预期影响、实施难度。
4. 给出“数据质量与下一步”建议（需要补充的口径或控制项）。
5. 不杜撰不存在的数据；如依据不足，请明确说明“需补充数据”。

请直接输出Markdown格式，无需再重复题目。
""".strip()


def generate_narrative(
    df_units: pd.DataFrame,
    df_stream: Optional[pd.DataFrame] = None,
    llm_params: Optional[dict] = None,
) -> str:
    ctx = _build_context(df_units, df_stream)
    prompt = _prompt_zh(ctx)
    llm_params = llm_params or {}
    return generate_with_llm(prompt, **llm_params)
