
import numpy as np
from models.company import CompanyModel

def build_linear_system(model: CompanyModel):
    produced = sorted(list(model.produced_material_ids()))
    idx = {m:i for i,m in enumerate(produced)}
    n = len(produced)
    A = np.zeros((n,n), dtype=float)
    b = np.zeros((n,), dtype=float)
    for mp in produced:
        r = idx[mp]
        Qp = 0.0
        for u in model.units.values():
            if mp in u.production:
                Qp += float(u.production[mp])
        A[r,r] += Qp
        for u in model.units.values():
            if mp not in u.production: continue
            for mi, q in u.consumption.items():
                if mi in model.fixed_factors:
                    b[r] += q * model.fixed_factors[mi]
                elif mi in idx:
                    A[r, idx[mi]] -= q
                else:
                    b[r] += q * model.carry_init.get(mi, 0.0)
    return A, b, produced

def solve_linear(model: CompanyModel):
    A, b, produced = build_linear_system(model)
    try:
        x = np.linalg.solve(A,b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(A,b, rcond=None)
    return {m: float(v) for m,v in zip(produced, x)}
