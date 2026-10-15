"""Small, generic I/O helpers shared across ETL scripts."""

from pathlib import Path


def find_column(columns, expected):
    """Case-insensitive column lookup. Returns None if no match is found."""
    return next(
        (col for col in columns if col.lower() == expected.lower()),
        None,
    )


def require_file(path, description):
    """Validate that an expected input file exists, with a clear error if not."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def require_columns(df, required, description):
    """Validate that a DataFrame has all required columns, with a clear error if not."""
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"{description} is missing required columns: {missing}. "
            f"Available columns: {sorted(df.columns)}"
        )
