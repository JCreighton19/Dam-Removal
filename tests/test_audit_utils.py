"""Checks for analysis/audit_utils.py (helpers of notebooks/preprocessing_audit.ipynb).

Robust z-scores on tiny in-memory tables (MAD, MeanAD fallback, constant,
sparse and not-evaluable groups), unit-factor matching, name matching, the
DO saturation formula, the output-folder guard and the read-only VAA reader.

Run directly: `python tests/test_audit_utils.py`
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import pandas as pd
import pyogrio

from analysis import audit_utils as au


def test_robust_z_methods():
    con = duckdb.connect()
    rows = (
        # MAD group: median 10, MAD 1 -> z of 20 is 10 / 1.4826
        [("A", v, False) for v in [9, 10, 11, 9, 10, 11, 10, 20]]
        # MAD 0 but not constant: MeanAD fallback
        + [("B", v, False) for v in [5, 5, 5, 5, 5, 5, 9]]
        # Constant group
        + [("C", 3, False)] * 6
        # Sparse: 2 reference values; the censored row doesn't count toward n
        + [("D", 1, False), ("D", 2, False), ("D", 0.5, True)]
        # Not evaluable: x null
        + [("A", None, False)]
    )
    con.register("src", pd.DataFrame(rows, columns=["g", "x", "is_censored"]))
    au.robust_z(con, "SELECT * FROM src", ["g"], "out", min_n=5, reference_filter="NOT is_censored")
    out = con.sql("SELECT * FROM out").df()

    a20 = out[(out["g"] == "A") & (out["x"] == 20)].iloc[0]
    assert a20["method"] == "mad"
    assert abs(a20["robust_z"] - 10 / 1.4826) < 1e-9
    b9 = out[(out["g"] == "B") & (out["x"] == 9)].iloc[0]
    assert b9["method"] == "meanad" and b9["robust_z"] > 0
    assert set(out.loc[out["g"] == "C", "method"]) == {"zero_spread"}
    assert out.loc[out["g"] == "C", "robust_z"].isna().all()
    assert set(out.loc[out["g"] == "D", "method"]) == {"sparse"}
    assert out.loc[out["x"].isna(), "method"].tolist() == ["not_evaluable"]
    assert len(out) == len(rows)


def test_nearest_factor():
    assert au.nearest_factor(1000) == "x1000 (mg vs ug)"
    assert au.nearest_factor(1100) == "x1000 (mg vs ug)"
    assert au.nearest_factor(0.001) == "/1000 (ug vs mg)"
    assert au.nearest_factor(4.43) == "NO3 vs N (x4.43)"
    assert au.nearest_factor(1.0) is None
    assert au.nearest_factor(2.0) is None
    assert au.nearest_factor(None) is None
    assert au.nearest_factor(-3) is None
    only_mass = {k: au.UNIT_FACTORS[k] for k in ["x1000 (mg vs ug)"]}
    assert au.nearest_factor(3.07, factors=only_mass) is None
    # Per-factor tolerances: 1200 is 20% off x1000, inside a 50% band only.
    wide = {label: (0.5 if "1000" in label else 0.15) for label in au.UNIT_FACTORS}
    assert au.nearest_factor(1200) is None
    assert au.nearest_factor(1200, tolerance=wide) == "x1000 (mg vs ug)"
    assert au.nearest_factor(3.8, tolerance=wide) is None  # between PO4 (3.07) and NO3 (4.43)


def test_name_matches():
    assert au.name_matches("SUCK CR TRIB NR ZION GROVE, NC", "Suck Creek") is True
    assert au.name_matches("Mill Creek at Route 9", "Otter River") is False
    assert au.name_matches("Anything", None) is None
    assert au.name_matches("Anything", "Creek") is None  # nothing distinctive to compare


def test_do_saturation():
    # Standard Methods table: about 9.09 mg/L at 20 C and 14.62 at 0 C.
    assert abs(au.do_saturation_mg_l(20) - 9.09) < 0.02
    assert abs(au.do_saturation_mg_l(0) - 14.62) < 0.02
    con = duckdb.connect()
    sql_value = con.sql(f"SELECT {au.do_saturation_sql('20.0')}").fetchone()[0]
    assert abs(sql_value - au.do_saturation_mg_l(20)) < 1e-9


def test_haversine():
    con = duckdb.connect()
    d = con.sql(f"SELECT {au.haversine_m_sql('0.0', '0.0', '0.0', '1.0')}").fetchone()[0]
    assert abs(d - 111195) < 50  # one degree of longitude at the equator


def test_save_output_stays_in_folder():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "audit"
        path = au.save_output(pd.DataFrame({"a": [1]}), "x.csv", out)
        assert path.exists() and not path.with_name("x.csv.tmp").exists()
        try:
            au.save_output(pd.DataFrame({"a": [1]}), "../escape.csv", out)
        except ValueError:
            pass
        else:
            raise AssertionError("wrote outside the output folder")
        assert not (Path(tmp) / "escape.csv").exists()


def test_counts():
    con = duckdb.connect()
    con.register("t", pd.DataFrame({
        "measurement_id": [1, 1, 2], "AR_ID": ["D1", "D2", "D1"], "station_id": ["S", "S", "S"],
        "parameter_std": ["ph", "ph", "phosphorus"], "sample_fraction_std": [None, None, "Total"],
    }))
    c = au.counts(con, "SELECT * FROM t")
    assert c == {"rows": 3, "measurements": 2, "stations": 1, "dams": 2, "pairs": 2, "series": 2}


def test_read_vaa_gpkg_and_missing_columns():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pkg.gpkg"
        # Lowercase names and float IDs, as in newer NHDPlus HR releases.
        vaa = pd.DataFrame({"nhdplusid": [10.0, 20.0, 30.0], "totdasqkm": [1.5, 2.5, -9999.0],
                            "slope": [0.01, -9998.0, 0.002]})
        pyogrio.write_dataframe(vaa, path, layer="NHDPlusFlowlineVAA", driver="GPKG")
        frame, notes = au.read_vaa([path], [10, 30, 99], ["NHDPlusID", "TotDASqKm", "Slope", "StreamOrde"])
        assert sorted(frame["nhdplusid"].tolist()) == [10, 30]
        assert frame.set_index("nhdplusid").loc[30, "totdasqkm"] == -9999.0
        assert any("streamorde" in n for n in notes)
        assert set(frame["package"]) == {"pkg"}


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")
