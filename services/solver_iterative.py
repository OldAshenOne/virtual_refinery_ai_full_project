
from models.company import CompanyModel
def solve_iterative(model: CompanyModel, max_iter=1000, tol=1e-8, verbose=False):
    produced = sorted(list(model.produced_material_ids()))
    x = {m: model.carry_init.get(m, 0.0) for m in produced}
    for it in range(max_iter):
        x_old = x.copy()
        for mp in produced:
            Qp = 0.0
            total_in = 0.0
            for u in model.units.values():
                if mp not in u.production: continue
                Qp += float(u.production[mp])
                for mi, q in u.consumption.items():
                    if mi in model.fixed_factors:
                        total_in += q * model.fixed_factors[mi]
                    elif mi in x:
                        total_in += q * x[mi]
                    else:
                        total_in += q * model.carry_init.get(mi, 0.0)
            if Qp > 0:
                x[mp] = total_in / Qp
        err = max(abs(x[m]-x_old.get(m,0.0)) for m in produced) if produced else 0.0
        if verbose and it % 10 == 0:
            print(f"iter {it}, err={err:.6e}")
        if err < tol:
            if verbose: print("converged at iter", it)
            break
    return x
