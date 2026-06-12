# FILE: app/vehicle/dtc_codes.py
# Purpose: Static descriptions for common OBD2 powertrain DTCs (diesel/Stellantis
#          weighted). Unknown codes return None — Astra explains them in chat.
# Called-by: app.vehicle.service (record_dtcs)
# Depends-on: stdlib only
# Last-renovated: 2026-06-12

COMMON_DTCS: dict[str, str] = {
    # Fuel / air metering (diesel)
    "P0087": "Fuel rail pressure too low",
    "P0088": "Fuel rail pressure too high",
    "P0090": "Fuel pressure regulator control circuit",
    "P0093": "Fuel system large leak detected",
    "P0100": "Mass air flow (MAF) circuit",
    "P0101": "MAF sensor range/performance",
    "P0102": "MAF sensor low input",
    "P0110": "Intake air temperature sensor circuit",
    "P0115": "Engine coolant temperature sensor circuit",
    "P0116": "Coolant temperature sensor range/performance",
    "P0120": "Throttle/pedal position sensor circuit",
    "P0180": "Fuel temperature sensor circuit",
    "P0190": "Fuel rail pressure sensor circuit",
    "P0191": "Fuel rail pressure sensor range/performance",
    "P0201": "Injector circuit — cylinder 1",
    "P0202": "Injector circuit — cylinder 2",
    "P0203": "Injector circuit — cylinder 3",
    "P0204": "Injector circuit — cylinder 4",
    "P0234": "Turbo overboost condition",
    "P0299": "Turbo underboost (boost leak / actuator)",
    # Glow plugs (cold-start complaints on diesels)
    "P0380": "Glow plug circuit A",
    "P0381": "Glow plug indicator circuit",
    "P0670": "Glow plug module control circuit",
    "P0671": "Glow plug circuit — cylinder 1",
    "P0672": "Glow plug circuit — cylinder 2",
    "P0673": "Glow plug circuit — cylinder 3",
    "P0674": "Glow plug circuit — cylinder 4",
    # EGR / emissions
    "P0400": "EGR flow malfunction",
    "P0401": "EGR insufficient flow",
    "P0402": "EGR excessive flow",
    "P0403": "EGR control circuit",
    "P0404": "EGR control range/performance",
    "P0420": "Catalyst efficiency below threshold",
    "P0470": "Exhaust pressure sensor circuit",
    "P0471": "Exhaust pressure sensor range/performance",
    # DPF (the big one on a delivery diesel)
    "P1435": "DPF differential pressure sensor (PSA)",
    "P1490": "DPF additive system fault (PSA)",
    "P2002": "DPF efficiency below threshold",
    "P2003": "DPF efficiency below threshold (bank 2)",
    "P2031": "Exhaust gas temperature sensor circuit",
    "P2080": "Exhaust gas temperature sensor range/performance",
    "P2452": "DPF pressure sensor circuit",
    "P2453": "DPF pressure sensor range/performance",
    "P2458": "DPF regeneration duration fault",
    "P2459": "DPF regeneration frequency fault",
    "P2463": "DPF soot accumulation excessive",
    # AdBlue / SCR (Euro 6 vans)
    "P204F": "SCR (AdBlue) system performance",
    "P20E8": "AdBlue pressure too low",
    "P20EE": "SCR NOx catalyst efficiency below threshold",
    "P2BAD": "NOx exceedance — AdBlue quality/dosing",
    # Charging / electrical
    "P0560": "System voltage malfunction",
    "P0562": "System voltage low (battery/alternator)",
    "P0563": "System voltage high",
    "P0606": "ECU processor fault",
    "P0615": "Starter relay circuit",
    # Transmission / driveline (common generic)
    "P0700": "Transmission control system malfunction",
    "P0730": "Incorrect gear ratio",
    # Misfire family (mostly petrol but generic)
    "P0300": "Random/multiple cylinder misfire",
    "P0301": "Cylinder 1 misfire",
    "P0302": "Cylinder 2 misfire",
    "P0303": "Cylinder 3 misfire",
    "P0304": "Cylinder 4 misfire",
}


def describe(code: str) -> str | None:
    """Description for a code, or None if unknown (raw code still shown)."""
    return COMMON_DTCS.get(str(code).strip().upper())
