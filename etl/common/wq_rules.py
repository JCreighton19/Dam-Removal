"""WQP cleaning rules for the wq_measurement table (docs/warehouse_spec.md, 3.6).

Kept as data, separate from clean_wq_results.py, so a new parameter, unit
or code found when the case study grows is a one-line change here. Unit
strings are matched after `normalize_unit` (lowercase, no whitespace).
"""

from __future__ import annotations

# --------------------------------------------------
# 3.6.1 Parameter mapping and standard units
# --------------------------------------------------

# CharacteristicName -> parameter_std. Dissolved oxygen rows in a percent
# unit are re-mapped to "do_saturation" once their unit is known.
PARAMETERS = {
    "Temperature, water": "water_temp",
    "Specific conductance": "specific_conductance",
    "Dissolved oxygen (DO)": "dissolved_oxygen",
    "Oxygen": "dissolved_oxygen",
    "pH": "ph",
    "Turbidity": "turbidity",
    "Nitrate": "nitrate",
    "Ammonia and ammonium": "ammonia",
    "Orthophosphate": "orthophosphate",
    "Phosphorus": "phosphorus",
    "Total dissolved solids": "tds",
}

DO_SATURATION_UNITS = {"%", "%saturatn"}

STANDARD_UNITS = {
    "water_temp": "deg C",
    "specific_conductance": "uS/cm @25C",
    "dissolved_oxygen": "mg/L",
    "do_saturation": "%",
    "ph": "std units",
    "turbidity": "NTU-equivalent",
    "nitrate": "mg/L as N",
    "ammonia": "mg/L as N",
    "orthophosphate": "mg/L as P",
    "phosphorus": "mg/L as P",
    "tds": "mg/L",
}

# No meaningful sample fraction: sample_fraction_std is null for these.
FIELD_PARAMETERS = {
    "water_temp", "specific_conductance", "dissolved_oxygen",
    "do_saturation", "ph", "turbidity",
}

FRACTION_ALIASES = {"Filtered, field": "Dissolved"}

# --------------------------------------------------
# 3.6.2 Row filters
# --------------------------------------------------

QC_ACTIVITY_PREFIX = "Quality Control"
KEPT_MEDIA = {"Water"}
EXCLUDED_MEDIA_SUBDIVISIONS = {"Bottom material", "Groundwater"}
EXCLUDED_STATUSES = {"Rejected"}
EXCLUDED_VALUE_TYPES = {"Calculated"}
SEDIMENT_FRACTIONS = {"Bed Sediment"}
SEDIMENT_UNITS = {"mg/kg"}
DROP_QUALIFIERS = {"H"}  # holding time exceeded

# NWIS alternate-form or derived codes for a measurement also reported
# under a kept code (as NH4 / NO3 / PO4, TDS loads, lab re-measurements).
DROP_PCODES = {
    "71846", "71845",           # ammonia as NH4
    "71851",                    # nitrate as NO3
    "00660",                    # orthophosphate as PO4
    "70301", "70302", "70303",  # TDS sum of constituents, tons/day, tons/acre-ft
    "90095",                    # specific conductance, lab
    "00403",                    # pH, lab
}

# Plausible ranges in result_unit_std (filter 10). Wide on purpose: they
# catch unit and entry errors, not unusual but real readings.
PLAUSIBLE_RANGES = {
    "water_temp": (-1.0, 40.0),
    "specific_conductance": (0.0, 60000.0),
    "dissolved_oxygen": (0.0, 25.0),
    "do_saturation": (0.0, 300.0),
    "ph": (0.0, 14.0),
    "turbidity": (0.0, 10000.0),
    "nitrate": (0.0, 100.0),
    "ammonia": (0.0, 100.0),
    "orthophosphate": (0.0, 50.0),
    "phosphorus": (0.0, 50.0),
    "tds": (0.0, 70000.0),
}

# --------------------------------------------------
# 3.6.3 Non-detects
# --------------------------------------------------

CENSORED_CONDITIONS = {
    "Not Detected",
    "Present Below Quantification Limit",
    "Below Reporting Limit",
}
CENSORED_VALUES = {"## (Censored)"}
CENSORED_QUALIFIERS = {"K"}

# --------------------------------------------------
# 3.6.4 Units
# --------------------------------------------------

# Unit implied by a kept NWIS parameter code, used when a row has no unit.
PCODE_UNITS = {
    "00010": "deg C",
    "00095": "uS/cm @25C",
    "00300": "mg/l",
    "00301": "% saturatn",
    "00400": "std units",
    "00608": "mg/l as N",
    "00610": "mg/l as N",
    "00618": "mg/l as N",
    "00665": "mg/l as P",
    "00666": "mg/l as P",
    "00671": "mg/l as P",
    "63676": "NTRU",
    "63680": "FNU",
    "70300": "mg/l",
}

# STORET reports pH without a unit.
DEFAULT_UNITS = {"ph": "std units"}

_N_NUTRIENTS = ("nitrate", "ammonia")
_P_NUTRIENTS = ("orthophosphate", "phosphorus")

# MethodSpeciationName (resultPhysChem profile) -> (basis, factor to that
# basis). Resolves a bare mg/L nutrient's chemical form instead of assuming
# it; only applied when the basis matches the parameter's standard unit.
SPECIATION_FACTORS = {
    "as N": ("N", 1.0),
    "as NO3": ("N", 14.007 / 62.004),
    "as NH4": ("N", 14.007 / 18.039),
    "as P": ("P", 1.0),
    "as PO4": ("P", 30.974 / 94.971),
}
NUTRIENT_BASIS = {**{p: "N" for p in _N_NUTRIENTS}, **{p: "P" for p in _P_NUTRIENTS}}


def _build_conversions():
    """(parameter_std, normalized unit) -> (factor, unit_basis_assumed)."""
    conversions = {}

    for parameter in ("dissolved_oxygen", "tds", *_N_NUTRIENTS, *_P_NUTRIENTS):
        # A bare mg/L nutrient doesn't say as N / as P: assumed and flagged.
        assumed = parameter in _N_NUTRIENTS + _P_NUTRIENTS
        conversions[(parameter, "mg/l")] = (1.0, assumed)
        conversions[(parameter, "ug/l")] = (0.001, assumed)

    for parameter in _N_NUTRIENTS:
        conversions[(parameter, "mg/lasn")] = (1.0, False)
        conversions[(parameter, "ug/lasn")] = (0.001, False)
    for parameter in _P_NUTRIENTS:
        conversions[(parameter, "mg/lasp")] = (1.0, False)
        conversions[(parameter, "ug/lasp")] = (0.001, False)

    for unit in ("us/cm", "umho/cm", "us/cm@25c"):
        conversions[("specific_conductance", unit)] = (1.0, False)
    for unit in DO_SATURATION_UNITS:
        conversions[("do_saturation", unit)] = (1.0, False)
    # NTU, NTRU and FNU have no conversion; treated as comparable (3.6.4).
    for unit in ("ntu", "ntru", "fnu"):
        conversions[("turbidity", unit)] = (1.0, False)

    conversions[("water_temp", "degc")] = (1.0, False)
    conversions[("ph", "stdunits")] = (1.0, False)
    return conversions


CONVERSIONS = _build_conversions()

# Specific conductance should be referenced to 25 deg C; other bases are logged.
CONDUCTANCE_BASES = {"25degc"}

# Depth unit -> meters.
DEPTH_TO_METERS = {"m": 1.0, "meters": 1.0, "ft": 0.3048, "feet": 0.3048, "cm": 0.01, "in": 0.0254}

# --------------------------------------------------
# 3.6.5 Deduplication
# --------------------------------------------------

# When one measurement is filed under two activities, keep the first type here.
ACTIVITY_PREFERENCE = ["Sample-Routine", "Field Msr/Obs"]


def normalize_unit(series):
    """Lowercase with all whitespace removed: "mg/l as N" -> "mg/lasn"."""
    return series.str.lower().str.replace(r"\s+", "", regex=True)


def normalize_pcode(series):
    """USGS parameter codes as 5-character strings ("608.0" -> "00608")."""
    text = series.str.strip().str.replace(r"\.0+$", "", regex=True)
    return text.where(~text.str.fullmatch(r"\d+").fillna(False), text.str.zfill(5))
