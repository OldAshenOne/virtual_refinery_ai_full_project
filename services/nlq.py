import pandas as pd
def answer(question: str, df_units: pd.DataFrame, df_stream: pd.DataFrame) -> str:
    q = question.lower()
    if "energy" in q or "co2" in q:
        for name in df_units["unit_name"]:
            if name.lower() in q:
                row = df_units[df_units["unit_name"]==name].iloc[0]
                return (f"{name} energy (inputs CO2e): {row['total_input_kgco2e']:.0f} kgCO2e; " 
                        f"output qty: {row['total_output_qty']:.0f}")
        top = df_units.sort_values("total_input_kgco2e", ascending=False).head(3)
        return "Top energy units:\n" + "\n".join(
            f"- {r['unit_name']}: {r['total_input_kgco2e']:.0f} kgCO2e" for _, r in top.iterrows()
        )
    if "stream" in q or "source" in q:
        top = df_stream.sort_values("factor_stream", ascending=False).head(5)
        return "Top streams by factor:\n" + "\n".join(
            f"- {r['stream_id']}: {r['factor_stream']:.4f}" for _, r in top.iterrows()
        )
    return "I can answer energy or stream questions. Try: 'FCC energy?' or 'Top streams by factor?'"
