# Runs reverse geocoding for ALL rows using Nominatim (OSM)
import time
import math
import re
import unicodedata
import random
import pandas as pd
import multiprocessing as mp

# CONFG
INPUT_FILE = r"C:\Users\rakes\Music\cities\CITES_Data\r_code\cites.cops.xlsx"
OUT_FILE   = r"C:\Users\rakes\Music\cities\CITES_Data\r_code\cites.cops.xlsx"

LAT_COL = "Latitude"
LON_COL = "Longitude"
AFF_COL = "Affiliation-internal"

# checkpoint write frequency (rewrites the Excel each time)
# For big files, 50 is very slow because Excel gets rewritten many times.
CHECKPOINT_EVERY = 500

RUN_nominatim = True

# Nominatim settings (be polite)
NOMINATIM_ZOOM = 10
NOMINATIM_SLEEP_SEC = 1.5  # sleep after each REAL API call (not cache hits)
CONTACT_EMAIL = "" #removed

# pip install reverse_geocoder pycountry requests openpyxl certifi
import reverse_geocoder as rg
import pycountry
import requests
import certifi


# Basic helpers
def to_float(x):
    try:
        if pd.isna(x):
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def norm(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = strip_accents(str(s)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # common aliases
    s = re.sub(r"\bberne\b", "bern", s)
    s = re.sub(r"\bgeneve\b", "geneva", s)
    s = re.sub(r"\bbruxelles\b", "brussels", s)
    s = re.sub(r"\bkobenhavn\b", "copenhagen", s)
    s = re.sub(r"\busa\b|\bu s a\b|\bus\b|\bu s\b", "united states", s)
    s = re.sub(r"\buk\b|\bu k\b", "united kingdom", s)
    return s


def token_in_aff(token: str, aff: str) -> bool:
    if not token:
        return False
    return re.search(rf"\b{re.escape(norm(token))}\b", norm(aff)) is not None


def country_name_from_cc(cc: str):
    if not cc:
        return None
    c = pycountry.countries.get(alpha_2=str(cc).upper())
    return c.name if c else None


def format_admin(city, state, country):
    parts = [p.strip() for p in [city, state, country] if isinstance(p, str) and p.strip()]
    return ", ".join(parts) if parts else None


def status_level(city, state, country):
    if city and str(city).strip():
        return "City"
    if state and str(state).strip():
        return "State"
    if country and str(country).strip():
        return "Country"
    return "Unknown"


def parse_geo_admin(x):
    if x is None or (isinstance(x, float) and math.isnan(x)) or str(x).strip() == "":
        return None, None, None
    parts = [p.strip() for p in str(x).split(",") if p.strip()]
    city = parts[0] if len(parts) > 0 else None
    state = parts[1] if len(parts) > 1 else None
    country = ", ".join(parts[2:]) if len(parts) > 2 else None
    return city, state, country

_COUNTRY_NORM = []
for c in pycountry.countries:
    _COUNTRY_NORM.append((norm(c.name), c.name))
    if hasattr(c, "official_name"):
        _COUNTRY_NORM.append((norm(c.official_name), c.name))
    if hasattr(c, "common_name"):
        _COUNTRY_NORM.append((norm(c.common_name), c.name))

_COUNTRY_NORM += [
    ("united states", "United States"),
    ("united kingdom", "United Kingdom"),
    ("czech republic", "Czechia"),
]


def extract_all_countries(aff: str) -> set:
    t = norm(aff)
    found = set()
    for cn, orig in sorted(_COUNTRY_NORM, key=lambda x: -len(x[0])):
        if cn and re.search(rf"\b{re.escape(cn)}\b", t):
            found.add(orig)
    return found


def looks_like_embassy(aff: str) -> bool:
    t = norm(aff)
    return any(k in t for k in [
        "embassy", "consulate", "permanent mission", "representation",
        "high commission", "mission", "delegation"
    ])


# Verify location
def verify_location(aff: str, geo_admin: str):
    if geo_admin is None or str(geo_admin).strip() == "":
        return "MISSING_GEO", "geo_admin blank"

    geo_city, geo_state, geo_country = parse_geo_admin(geo_admin)

    if token_in_aff(geo_city, aff):
        return "OK_CITY", "geo city found in affiliation"

    countries_in_aff = extract_all_countries(aff)

    if geo_country and countries_in_aff:
        if geo_country not in countries_in_aff:
            if looks_like_embassy(aff) and len(countries_in_aff) >= 2:
                return "MULTI_COUNTRY_OK", f"embassy/mission with countries {sorted(countries_in_aff)}; geo={geo_country}"
            return "WRONG_COUNTRY", f"affiliation mentions {sorted(countries_in_aff)} but geo says {geo_country}"

    state_ok = token_in_aff(geo_state, aff)
    country_ok = token_in_aff(geo_country, aff) or (geo_country in countries_in_aff if geo_country else False)

    if state_ok and country_ok:
        return "NEAR_CITY", "same state+country; city differs (suburb/nearby)"

    if country_ok:
        return "OK_COUNTRY", "country matches/mentioned; city not present in affiliation"

    if not countries_in_aff:
        return "UNVERIFIABLE", "no clear country token in affiliation text"

    return "WRONG_CITY", "no city/state token match (within same/unknown country)"


def add_verification_columns(df_geo: pd.DataFrame, geo_col: str, prefix: str) -> pd.DataFrame:
    out = df_geo.copy()
    ver = out.apply(lambda r: verify_location(r[AFF_COL], r[geo_col]), axis=1)
    out[f"{prefix}_verify_status"] = ver.apply(lambda x: x[0])
    out[f"{prefix}_verify_reason"] = ver.apply(lambda x: x[1])

    def note(st):
        if st in ("WRONG_COUNTRY", "WRONG_CITY"):
            return "Mismatch; geo kept original. Review Affiliation-internal."
        if st == "MISSING_GEO":
            return "Missing geo (check lat/lon)."
        return ""
    out[f"{prefix}_geo_note"] = out[f"{prefix}_verify_status"].apply(note)
    return out


# Online (Nominatim) 
session = requests.Session()
session.headers.update({
    "User-Agent": f"cites-geo-admin-only/1.0 (contact: {CONTACT_EMAIL})",
    "Accept-Language": "en"
})

# cache ONLY successes (do NOT cache failures)
_m2_cache = {}  # (lat_r, lon_r, zoom) -> (city, state, country)


class NominatimHTTPError(Exception):
    def __init__(self, status_code, message):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


def nominatim_reverse_admin(lat_r: str, lon_r: str, zoom: int):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "jsonv2",
        "lat": lat_r,
        "lon": lon_r,
        "zoom": zoom,
        "addressdetails": 1
    }

    last_err = None
    for attempt in range(1, 7):
        try:
            r = session.get(url, params=params, timeout=45, verify=certifi.where())

            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra and ra.strip().isdigit() else (3.0 * attempt)
                time.sleep(wait)
                continue

            if r.status_code in (401, 403):
                raise NominatimHTTPError(r.status_code, r.text[:200])

            if r.status_code >= 400:
                raise NominatimHTTPError(r.status_code, r.text[:200])

            data = r.json()
            addr = data.get("address", {}) or {}

            city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("hamlet")
            state = addr.get("state") or addr.get("region")
            country = addr.get("country")

            return (city, state, country, r.status_code)

        except Exception as e:
            last_err = e
            time.sleep(1.0 * attempt)

    raise last_err


def nominatim(df, sleep_sec=NOMINATIM_SLEEP_SEC, zoom=NOMINATIM_ZOOM):
    geo_list = []
    lvl_list = []
    err_list = []
    http_list = []
    api_called_list = []

    api_calls = 0
    cache_hits = 0
    consecutive_block = 0

    for i, (lat, lon) in enumerate(zip(df[LAT_COL], df[LON_COL]), start=1):
        la = to_float(lat)
        lo = to_float(lon)

        if la is None or lo is None:
            geo_list.append(None)
            lvl_list.append("Unknown")
            err_list.append("missing lat/lon")
            http_list.append(None)
            api_called_list.append(False)
            continue

        lat_r = f"{la:.5f}"
        lon_r = f"{lo:.5f}"
        key = (lat_r, lon_r, zoom)

        if key in _m2_cache:
            city, state, country = _m2_cache[key]
            geo_list.append(format_admin(city, state, country))
            lvl_list.append(status_level(city, state, country))
            err_list.append("")
            http_list.append(200)
            api_called_list.append(False)
            cache_hits += 1
            continue

        api_called_list.append(True)
        api_calls += 1

        try:
            city, state, country, http_status = nominatim_reverse_admin(lat_r, lon_r, zoom)

            # cache success
            _m2_cache[key] = (city, state, country)

            geo_list.append(format_admin(city, state, country))
            lvl_list.append(status_level(city, state, country))
            err_list.append("")
            http_list.append(http_status)
            consecutive_block = 0

        except NominatimHTTPError as e:
            geo_list.append(None)
            lvl_list.append("Unknown")
            err_list.append(str(e))
            http_list.append(getattr(e, "status_code", None))

            if getattr(e, "status_code", None) in (401, 403):
                consecutive_block += 1
                if consecutive_block >= 2:
                    print("[M2 STOP] Blocked (401/403). Stopping online for remaining rows in this chunk.")
                    rem = len(df) - i
                    geo_list.extend([None] * rem)
                    lvl_list.extend(["Unknown"] * rem)
                    err_list.extend([str(e)] * rem)
                    http_list.extend([getattr(e, "status_code", None)] * rem)
                    api_called_list.extend([False] * rem)
                    break

        except Exception as e:
            geo_list.append(None)
            lvl_list.append("Unknown")
            err_list.append(str(e))
            http_list.append(None)

        # rate-limit friendly sleep after every real call attempt
        time.sleep(sleep_sec + random.uniform(0.0, 0.35))

    out = df.copy()
    out["geo_admin_m2"] = geo_list
    out["affiliation_status_m2"] = lvl_list
    out["m2_error"] = err_list
    out["m2_http_status"] = http_list
    out["m2_api_called"] = api_called_list

    print(f"[M2 DONE] api_calls={api_calls} cache_hits={cache_hits} (this chunk)")
    return out


# Writing
def make_counts(df: pd.DataFrame, status_col: str) -> pd.DataFrame:
    counts = df[status_col].value_counts(dropna=False).reset_index()
    counts.columns = ["verify_status", "count"]
    return counts


def write_checkpoint(out_path, full_m1, mism_m1, counts_m1, full_m2, mism_m2, counts_m2):
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        if full_m1 is not None:
            full_m1.to_excel(w, sheet_name="m1_all", index=False)
            (mism_m1 if mism_m1 is not None else full_m1.head(0)).to_excel(w, sheet_name="m1_mismatch", index=False)
            (counts_m1 if counts_m1 is not None else pd.DataFrame()).to_excel(w, sheet_name="m1_counts", index=False)

        if full_m2 is not None:
            full_m2.to_excel(w, sheet_name="m2_all", index=False)
            (mism_m2 if mism_m2 is not None else full_m2.head(0)).to_excel(w, sheet_name="m2_mismatch", index=False)
            (counts_m2 if counts_m2 is not None else pd.DataFrame()).to_excel(w, sheet_name="m2_counts", index=False)


def main():
    # ALL rows (no nrows limit)
    df = pd.read_excel(
        INPUT_FILE,
        engine="openpyxl",
        usecols=[AFF_COL, LAT_COL, LON_COL]
    )

    missing = [c for c in [AFF_COL, LAT_COL, LON_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    df["row_id"] = range(1, len(df) + 1)

    m1_chunks = []
    m2_chunks = []

    for start in range(0, len(df), CHECKPOINT_EVERY):
        end = min(start + CHECKPOINT_EVERY, len(df))
        chunk = df.iloc[start:end].copy()

        full_m1 = mism_m1 = counts_m1 = None
        full_m2 = mism_m2 = counts_m2 = None

        if RUN_nominatim:
            m2 = nominatim(chunk)
            m2 = add_verification_columns(m2, "geo_admin_m2", "m2")
            m2_chunks.append(m2)

            full_m2 = pd.concat(m2_chunks, ignore_index=True)
            mism_m2 = full_m2[full_m2["m2_verify_status"].isin(["WRONG_COUNTRY", "WRONG_CITY", "MISSING_GEO"])].copy()
            counts_m2 = make_counts(full_m2, "m2_verify_status")

        write_checkpoint(OUT_FILE, full_m1, mism_m1, counts_m1, full_m2, mism_m2, counts_m2)
        print(f"[CHECKPOINT] wrote rows 1..{end} -> {OUT_FILE}")

    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
