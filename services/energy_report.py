
import pandas as pd
def build_energy_summary(model) -> pd.DataFrame:
    rows = []
    for uid, u in model.units.items():
        elec = u.consumption.get("M_ELEC", 0.0)
        steam = u.consumption.get("M_STEAM", 0.0)
        fg = u.consumption.get("M_FUELGAS", 0.0)
        c_elec = elec * model.fixed_factors.get("M_ELEC", 0.0)
        c_steam = steam * model.fixed_factors.get("M_STEAM", 0.0)
        c_fg = fg * model.fixed_factors.get("M_FUELGAS", 0.0)
        rows.append(dict(unit_id=uid, unit_name=u.unit_name,
                         elec_kgco2e=c_elec, steam_kgco2e=c_steam,
                         fuelgas_kgco2e=c_fg, total_kgco2e=(c_elec+c_steam+c_fg)))
    return pd.DataFrame(rows)
