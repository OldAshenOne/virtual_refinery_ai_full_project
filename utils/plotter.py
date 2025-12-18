
import matplotlib.pyplot as plt
def save_energy_chart(df_energy, out_png: str):
    units = df_energy['unit_name'].tolist()
    totals = df_energy['total_kgco2e'].tolist()
    plt.figure(figsize=(10,6))
    plt.bar(units, totals)
    plt.title('Unit Energy CO2e (Electricity + Steam + Fuel Gas)')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('kgCO2e')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
# === Add: clustering & anomaly charts ===
def save_cluster_scatter(df_units_clustered, out_png: str):
    """
    散点：x=产量，总量；y=输入CO2e；点上标注装置名与簇号。
    """
    x = df_units_clustered["total_output_qty"].astype(float)
    y = df_units_clustered["total_input_kgco2e"].astype(float)
    labels = df_units_clustered["unit_name"].astype(str)
    clusters = df_units_clustered["cluster"].astype(str)

    plt.figure(figsize=(9,6))
    plt.scatter(x, y, s=60)  # 不指定颜色/样式
    for xi, yi, name, clu in zip(x, y, labels, clusters):
        plt.text(xi, yi, f"{name} (c{clu})", fontsize=8)
    plt.xlabel("Total output qty")
    plt.ylabel("Total input CO2e (kgCO2e)")
    plt.title("Unit clustering: output vs energy (labels show cluster id)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

def save_cluster_energy_bar(df_units_clustered, out_png: str):
    """
    柱状：各簇的平均输入CO2e。
    """
    g = (df_units_clustered
         .groupby("cluster", as_index=False)["total_input_kgco2e"]
         .mean()
         .sort_values("cluster"))
    plt.figure(figsize=(8,5))
    plt.bar(g["cluster"].astype(str), g["total_input_kgco2e"])
    plt.xlabel("Cluster id")
    plt.ylabel("Avg input CO2e (kgCO2e)")
    plt.title("Average energy (CO2e) by cluster")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

def save_anomaly_zscore_bar(df_units_anom, out_png: str, z_thresh: float = 2.0):
    """
    柱状：各装置按 z-score（能耗）排序，显示阈值线。
    需要 df_units_anom 中包含列 'z_energy'（detect_outliers 已生成）。
    """
    a = df_units_anom.copy()
    a = a.sort_values("z_energy")
    plt.figure(figsize=(10,6))
    plt.bar(a["unit_name"].astype(str), a["z_energy"].astype(float))
    plt.axhline(z_thresh, linestyle="--")
    plt.axhline(-z_thresh, linestyle="--")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("z-score (energy)")
    plt.title("Energy anomaly by z-score")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
