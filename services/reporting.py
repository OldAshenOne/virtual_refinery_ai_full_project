
import pandas as pd
def build_reports(model, f_lin: dict, f_it: dict, f_stream: dict):
    rows = []
    for uid, u in model.units.items():
        total_in = 0.0
        for mi, q in u.consumption.items():
            if mi in model.fixed_factors:
                f = model.fixed_factors[mi]
            elif mi in f_lin:
                f = f_lin[mi]
            elif mi in f_it:
                f = f_it[mi]
            else:
                f = model.carry_init.get(mi, 0.0)
            total_in += q * f
        total_out = sum(float(q) for q in u.production.values())
        rows.append(dict(unit_id=uid, unit_name=u.unit_name,
                         total_input_kgco2e=total_in, total_output_qty=total_out))
    df_units = pd.DataFrame(rows)
    df_lin = pd.DataFrame([dict(material_id=k, factor_linear=v) for k,v in f_lin.items()])
    df_it  = pd.DataFrame([dict(material_id=k, factor_iter=v) for k,v in f_it.items()])
    df_stream = pd.DataFrame([dict(stream_id=k, factor_stream=v) for k,v in f_stream.items()])
    return df_lin, df_it, df_stream, df_units
