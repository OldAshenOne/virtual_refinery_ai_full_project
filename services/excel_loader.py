
import pandas as pd
from models.material import Material
from models.unit import UnitData
from models.company import CompanyModel

def load_company_from_excel(file_path: str) -> CompanyModel:
    xls = pd.ExcelFile(file_path)
    materials = pd.read_excel(xls, "materials")
    fixed = pd.read_excel(xls, "fixed_factors")
    units = pd.read_excel(xls, "units")
    cons = pd.read_excel(xls, "consumption")
    prod = pd.read_excel(xls, "production")
    carry = pd.read_excel(xls, "carryover")
    try:
        routing = pd.read_excel(xls, "routing")
    except Exception:
        routing = pd.DataFrame(columns=["consumer_unit_id","material_id","source_unit_id","amount","unit"])

    model = CompanyModel("Virtual Refinery", 2025)

    for _, r in materials.iterrows():
        m = Material(r['material_id'], r['material_name'], r['unit'], r['category'])
        model.materials[m.material_id] = m

    for _, r in fixed.iterrows():
        model.fixed_factors[r['material_id']] = float(r['factor_per_unit'])

    for _, r in units.iterrows():
        model.units[r['unit_id']] = UnitData(r['unit_id'], r['unit_name'])

    for _, r in cons.iterrows():
        uid, mid, amt = r['unit_id'], r['material_id'], float(r['amount'])
        if amt < 0:
            raise ValueError(f"Negative consumption not supported: unit={uid}, material={mid}, amount={amt}")
        if uid not in model.units:
            model.units[uid] = UnitData(uid, uid)
        model.units[uid].consumption[mid] = model.units[uid].consumption.get(mid, 0.0) + amt

    for _, r in prod.iterrows():
        uid, mid, amt = r['unit_id'], r['material_id'], float(r['amount'])
        if amt < 0:
            raise ValueError(f"Negative production not supported: unit={uid}, material={mid}, amount={amt}")
        if uid not in model.units:
            model.units[uid] = UnitData(uid, uid)
        model.units[uid].production[mid] = model.units[uid].production.get(mid, 0.0) + amt

    for _, r in carry.iterrows():
        val = float(r['factor_init'])
        if val < 0:
            raise ValueError(f"Negative initial factor not supported: material={r['material_id']}, factor={val}")
        model.carry_init[r['material_id']] = val

    if not routing.empty:
        for _, r in routing.iterrows():
            amt = float(r['amount'])
            if amt < 0:
                raise ValueError(
                    f"Negative routed amount not supported: consumer={r['consumer_unit_id']}, material={r['material_id']}, source={r['source_unit_id']}, amount={amt}"
                )
            key = (r['consumer_unit_id'], r['material_id'])
            model.routing.setdefault(key, []).append((r['source_unit_id'], amt))

    return model
