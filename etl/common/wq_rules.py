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
    # Added 2026-11-02 (advisor: suspended sediment and stream flow). SSC and
    # TSS are kept apart: TSS subsamples the bottle and under-reads sand, so
    # the two aren't comparable (Gray et al. 2000).
    "Suspended Sediment Concentration (SSC)": "ssc",
    "Total suspended solids": "tss",
    # Discharge measured with a sample (USGS 00061; "Flow" from other
    # agencies). Daily means ("Stream flow, mean. daily") are a different
    # quantity and are left out; continuous gauge records come from NWIS,
    # not WQP.
    "Stream flow, instantaneous": "discharge",
    "Flow": "discharge",
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
    "ssc": "mg/L",
    "tss": "mg/L",
    "discharge": "ft3/s",
}

# No meaningful sample fraction: sample_fraction_std is null for these.
FIELD_PARAMETERS = {
    "water_temp", "specific_conductance", "dissolved_oxygen",
    "do_saturation", "ph", "turbidity", "discharge",
}
# TDS is dissolved and SSC / TSS suspended by definition, so their fraction
# labels carry no meaning.
NO_FRACTION_PARAMETERS = FIELD_PARAMETERS | {"tds", "ssc", "tss"}

# Agency labels for the same two fractions: filtered (Dissolved) or not (Total).
FRACTION_ALIASES = {
    "Filtered, field": "Dissolved",
    "Filtered, lab": "Dissolved",
    "Filtered field and/or lab": "Dissolved",
    "Unfiltered": "Total",
}

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
    # Floor 2 uS/cm (2026-11-08): below that is distilled-water conductance.
    # Values there are mS/cm reported as uS/cm (CHIPCREE_WQX from 2015, MNPCA
    # in 2015, KARUKDNR_WQX: ~0.4-0.7 at stations that read 350-900 in other
    # years; notebooks/qa/conductance_plausibility.ipynb) or zeros. Dropped,
    # not multiplied by 1000: the unit is inferred, not reported. Very dilute
    # real waters (3-5 uS/cm, e.g. Sierra Nevada snowmelt) are kept.
    "specific_conductance": (2.0, 60000.0),
    "dissolved_oxygen": (0.0, 25.0),
    "do_saturation": (0.0, 300.0),
    "ph": (0.0, 14.0),
    "turbidity": (0.0, 10000.0),
    "nitrate": (0.0, 100.0),
    "ammonia": (0.0, 100.0),
    "orthophosphate": (0.0, 50.0),
    "phosphorus": (0.0, 50.0),
    # Floor 2 mg/L (2026-11-08), like conductance: lower is not natural water.
    # Drops field-meter TDS reported in g/L but labelled mg/L (ONEIDA_WQX
    # 2007-09: 0.35-0.77 next to lab values of 380-970; TDS / conductance
    # 0.0006 instead of ~0.55-0.75).
    "tds": (2.0, 70000.0),
    # Hyperconcentrated flows reach several hundred thousand mg/L.
    "ssc": (0.0, 600000.0),
    "tss": (0.0, 600000.0),
    # Above the largest recorded U.S. river flows; negative (tidal) flows are
    # dropped, since the study is of rivers.
    "discharge": (0.0, 5000000.0),
}

# --------------------------------------------------
# 3.6.3 Non-detects
# --------------------------------------------------

CENSORED_CONDITIONS = {
    "Not Detected",
    "Present Below Quantification Limit",
    "Below Reporting Limit",
    # Added 2026-11-08 (notebooks/qa/conductance_plausibility.ipynb section 8):
    # a wording of "Not Detected" the list missed (seen on TSS and phosphorus zeros).
    "Not Detected at Detection Limit",
}
CENSORED_VALUES = {"## (Censored)"}
# Matched exactly after trimming and upper-casing each code (codes are split
# on ";"). U, DL, BRL, BDL, ND added 2026-11-08: standard not-detected /
# below-limit codes that were reaching the warehouse as detected values (often
# as 0, which a log scale turns into an extreme value).
CENSORED_QUALIFIERS = {"K", "U", "DL", "BRL", "BDL", "ND"}

# Log-scale outcome measures (the log_parameters of analysis/mixed_model.py,
# minus discharge). An uncensored 0 for one of these is a "below what the kit
# or meter can read" report, not a concentration: it is treated as a
# non-detect (half a valid reported detection limit, otherwise dropped).
# Discharge is left out on purpose: zero flow is real. (2026-11-08)
ZERO_AS_NONDETECT = frozenset({
    "turbidity", "specific_conductance", "nitrate", "ammonia", "orthophosphate",
    "phosphorus", "tds", "ssc", "tss",
})

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
    "80154": "mg/l",     # SSC
    "00530": "mg/l",     # TSS
    "00061": "ft3/s",    # discharge, instantaneous
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

    for parameter in ("dissolved_oxygen", "tds", "ssc", "tss", *_N_NUTRIENTS, *_P_NUTRIENTS):
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

    # Discharge to ft3/s.
    for unit in ("ft3/s", "ft3/sec", "cfs"):
        conversions[("discharge", unit)] = (1.0, False)
    for unit in ("m3/s", "m3/sec", "cms"):
        conversions[("discharge", unit)] = (35.3147, False)
    conversions[("discharge", "gpm")] = (0.0022280, False)   # US gal / min
    conversions[("discharge", "mgd")] = (1.547229, False)    # million US gal / day

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
