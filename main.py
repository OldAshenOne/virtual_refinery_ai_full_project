import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from services.excel_loader import load_company_from_excel
from services.solver_linear import solve_linear
from services.solver_iterative import solve_iterative
from services.solver_stream import solve_stream_factors
from services.reporting import build_reports
from services.export_service import export_results
from services.energy_report import build_energy_summary
from utils.plotter import save_energy_chart, save_cluster_scatter, save_cluster_energy_bar, save_anomaly_zscore_bar
from services.ai_report import generate_narrative
from services.nlq import answer
import importlib.util, types
spec_clu = importlib.util.spec_from_file_location('clustering', os.path.join(BASE_DIR,'analytics','clustering.py'))
clustering = importlib.util.module_from_spec(spec_clu);
spec_clu.loader.exec_module(clustering)
cluster_units = clustering.cluster_units
spec_an = importlib.util.spec_from_file_location('anomaly', os.path.join(BASE_DIR,'analytics','anomaly.py'))
anomaly = importlib.util.module_from_spec(spec_an);
spec_an.loader.exec_module(anomaly)
detect_outliers = anomaly.detect_outliers
spec_sc = importlib.util.spec_from_file_location('scenario', os.path.join(BASE_DIR,'analytics','scenario.py'))
scenario = importlib.util.module_from_spec(spec_sc);
spec_sc.loader.exec_module(scenario)
what_if_change_fixed_factor = scenario.what_if_change_fixed_factor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
in_xlsx = os.path.join(BASE_DIR, "..", "virtual_refinery_data.xlsx")
out_xlsx = os.path.join(BASE_DIR, "..", "virtual_refinery_results.xlsx")
out_png  = os.path.join(BASE_DIR, "..", "unit_energy_co2e.png")
if __name__ == "__main__":
    print("Reading:", in_xlsx)
    model = load_company_from_excel(in_xlsx)
    f_lin = solve_linear(model)
    f_it  = solve_iterative(model, verbose=True)
    f_stream = solve_stream_factors(model)
    df_lin, df_it, df_stream, df_units = build_reports(model, f_lin, f_it, f_stream)
    export_results(df_lin, df_it, df_units, out_xlsx, df_stream=df_stream)
    df_energy = build_energy_summary(model)
    save_energy_chart(df_energy, out_png)
    narrative = generate_narrative(df_units, df_stream)
    with open(os.path.join(BASE_DIR, "..", "narrative.txt"), "w", encoding="utf-8") as f:
        f.write(narrative)
    df_units_clustered = cluster_units(df_units, n_clusters=3)
    df_units_clustered.to_excel(os.path.join(BASE_DIR, "..", "units_clustered.xlsx"), index=False)
    df_units_anom = detect_outliers(df_units)
    df_units_anom.to_excel(os.path.join(BASE_DIR, "..", "units_anomaly.xlsx"), index=False)
    print(answer("FCC energy?", df_units, df_stream))
    elec_old = model.fixed_factors.get("M_ELEC", 0.0)
    _, df_units_new = what_if_change_fixed_factor(model, "M_ELEC", elec_old * 1.2)
    df_units_new.to_excel(os.path.join(BASE_DIR, "..", "units_whatif_elec_plus20.xlsx"), index=False)
    print("Wrote results to:", out_xlsx)
    print("Saved energy chart to:", out_png)
    print("Saved narrative.txt, units_clustered.xlsx, units_anomaly.xlsx, units_whatif_elec_plus20.xlsx")

    
    # --- visuals for clustering & anomaly ---
    cluster_scatter_png = os.path.join(BASE_DIR, "..", "cluster_scatter.png")
    cluster_energy_bar_png = os.path.join(BASE_DIR, "..", "cluster_mean_energy.png")
    anomaly_bar_png = os.path.join(BASE_DIR, "..", "anomaly_zscore.png")

    save_cluster_scatter(df_units_clustered, cluster_scatter_png)
    save_cluster_energy_bar(df_units_clustered, cluster_energy_bar_png)
    save_anomaly_zscore_bar(df_units_anom, anomaly_bar_png, z_thresh=2.0)

    print("Saved clustering charts to:", cluster_scatter_png, "and", cluster_energy_bar_png)
    print("Saved anomaly chart to:", anomaly_bar_png)
