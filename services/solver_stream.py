
import numpy as np
from config import DEFAULT_ALLOCATION_METHOD
STREAM_SEP = "@"
def stream_id(material_id: str, unit_id: str) -> str:
    return f"{material_id}{STREAM_SEP}{unit_id}"

    
def build_stream_system(model):
    produced_streams = []
    for uid, u in model.units.items():
        for mid in u.production.keys():
            produced_streams.append(stream_id(mid, uid))
    produced_streams = sorted(set(produced_streams))
    idx = {s:i for i,s in enumerate(produced_streams)}
    n = len(produced_streams)
    A = np.zeros((n,n), dtype=float)
    b = np.zeros((n,), dtype=float)
    mat_producers = {}
    for uid, u in model.units.items():
        for mid in u.production.keys():
            mat_producers.setdefault(mid, []).append(uid)
    for s in produced_streams:
        mid_p, uid_p = s.split(STREAM_SEP, 1)
        row = idx[s]
        Qp = float(model.units[uid_p].production.get(mid_p, 0.0))
        A[row, row] += Qp
        unit = model.units[uid_p]
        for mi, q in unit.consumption.items():
            if mi in model.fixed_factors:
                b[row] += q * model.fixed_factors[mi]
                continue
            key = (uid_p, mi)
            if hasattr(model, "routing") and key in model.routing and len(model.routing[key])>0:
                amount_sum = 0.0
                for src_uid, q_sub in model.routing[key]:
                    amount_sum += q_sub
                    s_in = stream_id(mi, src_uid)
                    if s_in in idx:
                        A[row, idx[s_in]] -= q_sub
                    else:
                        b[row] += q_sub * model.carry_init.get(mi, 0.0)
                delta = q - amount_sum
                if abs(delta) > 1e-9:
                    producers = mat_producers.get(mi, [])
                    if len(producers)==1:
                        s_in = stream_id(mi, producers[0])
                        if s_in in idx:
                            A[row, idx[s_in]] -= delta
                        else:
                            b[row] += delta * model.carry_init.get(mi, 0.0)
                    else:
                        b[row] += delta * model.carry_init.get(mi, 0.0)
                continue
            producers = mat_producers.get(mi, [])
            if len(producers) == 1:
                s_in = stream_id(mi, producers[0])
                if s_in in idx:
                    A[row, idx[s_in]] -= q
                else:
                    b[row] += q * model.carry_init.get(mi, 0.0)
                continue
            # Multiple potential producers: apply average allocation by default
            if len(producers) > 1 and DEFAULT_ALLOCATION_METHOD == "average":
                share = q / float(len(producers)) if producers else 0.0
                for p_uid in producers:
                    s_in = stream_id(mi, p_uid)
                    if s_in in idx:
                        A[row, idx[s_in]] -= share
                    else:
                        b[row] += share * model.carry_init.get(mi, 0.0)
                continue
            # Fallback to carry_init if no clear producer mapping
            b[row] += q * model.carry_init.get(mi, 0.0)
    return A, b, produced_streams
def solve_stream_factors(model):
    A, b, streams = build_stream_system(model)
    try:
        x = np.linalg.solve(A,b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(A,b, rcond=None)
    return {s: float(v) for s,v in zip(streams, x)}
