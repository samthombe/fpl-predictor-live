# news_parser.py -- INJURY / AVAILABILITY AWARENESS (authoritative FPL feed)
import pandas as pd
from datetime import datetime, timezone

STATUS_LABEL = {
    "a": "Available", "d": "Doubtful", "i": "Injured",
    "s": "Suspended", "u": "Unavailable", "n": "Not in squad",
}
OUT_STATUSES = {"i", "s", "u"}


def _news_age_hours(news_added):
    if not news_added:
        return None
    try:
        dt = pd.to_datetime(news_added, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return (datetime.now(timezone.utc) - dt.to_pydatetime()).total_seconds() / 3600.0
    except Exception:
        return None


def availability_report(players: pd.DataFrame):
    p = players.copy()
    if p.empty:
        return p, {"out": 0, "doubt": 0, "monitor": 0, "fresh24": 0, "fresh48": 0}

    for col, default in [("news", ""), ("news_added", ""), ("status", "a"), ("chance_pct", 100.0)]:
        if col not in p.columns:
            p[col] = default
    p["news"] = p["news"].fillna("").astype(str)
    p["news_added"] = p["news_added"].fillna("").astype(str)
    p["status"] = p["status"].fillna("a").astype(str).str.lower()
    p["chance_pct"] = pd.to_numeric(p["chance_pct"], errors="coerce").fillna(100.0)
    p["status_label"] = p["status"].map(STATUS_LABEL).fillna("Unknown")
    p["news_age_h"] = p["news_added"].map(_news_age_hours)

    def tier(r):
        if r["status"] in OUT_STATUSES or r["chance_pct"] <= 0:
            return "Out"
        if r["status"] == "d" or r["chance_pct"] < 75:
            return "Doubt"
        age = r["news_age_h"]
        if r["chance_pct"] < 100 or (r["news"] and age is not None and age <= 72):
            return "Monitor"
        return "Clear"

    p["risk_tier"] = p.apply(tier, axis=1)
    p["fresh"] = p["news_age_h"].map(lambda h: h is not None and h <= 24)

    flagged = p[p["risk_tier"] != "Clear"].copy()
    order = {"Out": 0, "Doubt": 1, "Monitor": 2}
    flagged["_o"] = flagged["risk_tier"].map(order)
    flagged = flagged.sort_values(["_o", "chance_pct"]).drop(columns="_o")

    summary = {
        "out": int((p["risk_tier"] == "Out").sum()),
        "doubt": int((p["risk_tier"] == "Doubt").sum()),
        "monitor": int((p["risk_tier"] == "Monitor").sum()),
        "fresh24": int(p["fresh"].sum()),
        "fresh48": int(p["news_age_h"].map(lambda h: h is not None and h <= 48).sum()),
    }
    return flagged, summary