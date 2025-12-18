
import pandas as pd
def export_results(df_lin, df_it, df_units, out_path, df_stream=None):
    with pd.ExcelWriter(out_path) as w:
        df_lin.to_excel(w, sheet_name="factors_linear", index=False)
        df_it.to_excel(w, sheet_name="factors_iterative", index=False)
        df_units.to_excel(w, sheet_name="units_summary", index=False)
        if df_stream is not None:
            df_stream.to_excel(w, sheet_name="factors_stream", index=False)
