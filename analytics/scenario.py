
from copy import deepcopy
from services.solver_linear import solve_linear
from services.reporting import build_reports
from services.solver_iterative import solve_iterative
from services.solver_stream import solve_stream_factors

def what_if_change_fixed_factor(model, material_id: str, new_factor: float):
    model2 = deepcopy(model)
    if material_id in model2.fixed_factors:
        model2.fixed_factors[material_id] = float(new_factor)
    f_lin_new = solve_linear(model2)
    f_it_new = solve_iterative(model2)
    f_stream_new = solve_stream_factors(model2)
    _, _, _, df_units_new = build_reports(model2, f_lin_new, f_it_new, f_stream_new)
    return f_lin_new, df_units_new
