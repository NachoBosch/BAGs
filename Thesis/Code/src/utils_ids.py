import re

_LEADING_ID_RE = re.compile(r"^([A-Z]{2,5}\d{2,6})")


def normalize_record_id(rid: str) -> str:
    """Normalize a record ID to its canonical form (e.g. 'AF025').

    Handles T1w filenames (cat_sub-AF025_ses-T0_T1w.xml), FC filenames
    (sub-AF025_timeseries.mat), and plain IDs (AF025, BE00128).
    """
    s = (rid or "").strip().upper()
    s = s.replace("CAT_SUB-", "").replace("SUB-", "")
    s = s.replace("-", "_").replace(".", "_")
    s = re.sub(r"_SES[_]?[A-Z0-9]+", "", s)
    s = re.sub(r"_RUN[_]?\d+", "", s)
    s = re.sub(r"_T1W.*", "", s)

    m = _LEADING_ID_RE.match(s)
    if m:
        return m.group(1)

    s = re.sub(r"(T1W|XML|MAT|NII)+$", "", s)
    m2 = _LEADING_ID_RE.match(s)
    return m2.group(1) if m2 else s
