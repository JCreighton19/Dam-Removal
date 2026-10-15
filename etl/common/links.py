"""Shared helpers for reading dam-station link tables produced by
link_dams_stations.py.
"""

from __future__ import annotations

import pandas as pd

from .io import require_columns

# A pair is "network-connected" if link_dams_stations.py classified it as
# one of these -- "unconnected" pairs are excluded wherever this is used.
CONNECTED_RELATIONSHIPS = ("upstream", "downstream", "same_reach")

REQUIRED_COLUMNS = {"dam_id", "dam_name", "station_id", "year_removed", "relationship"}


def load_links(path):
    df = pd.read_csv(path, dtype={"station_id": "string", "dam_id": "string"})
    require_columns(df, REQUIRED_COLUMNS, f"Dam-station link table ({path})")
    return df


def connected_only(links):
    """Filter a loaded link table down to network-connected rows."""
    return links[links["relationship"].isin(CONNECTED_RELATIONSHIPS)].copy()
