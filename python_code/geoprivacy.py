# -*- coding: utf-8 -*-
"""
Confirm geolocations are NOT too precise (no building / street-level).
Create a new field: geo_precision with values: city, state_province, country, unknown
Ensure published coordinates reflect that precision (use centroid of that admin unit).
Do NOT overwrite your original Latitude/Longitude
"""

import argparse
import json
import math
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable


# ----------------------------
# Helpers: haversine distance
# ----------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km; returns np.nan if any input missing."""
    try:
        if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
            return np.nan
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except Exception:
        return np.nan

    R = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# Cache load/save (permission-safe)
def load_cache(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {"reverse_safe": {}, "geocode_safe": {}}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    obj.setdefault("reverse_safe", {})
    obj.setdefault("geocode_safe", {})
    return obj


def save_cache(path: str, cache: Dict) -> None:
    """Atomic-ish save with Windows permission fallback."""
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # Try replace with a few retries (Windows file lock issues)
    for i in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25 * (i + 1))

    # Fallback: write to a new file so you don’t lose progress
    fallback = path.replace(".json", f".fallback.{int(time.time())}.json")
    with open(fallback, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"WARNING: Could not overwrite cache (locked). Wrote fallback cache: {fallback}")



CITY_KEYS = ["city", "town", "village", "municipality", "locality"]
REGION_KEYS = ["state", "province", "region"]
COUNTRY_KEYS = ["country"]

TOO_PRECISE_KEYS = ["house_number", "road", "pedestrian", "footway", "postcode", "amenity"]
SUBCITY_KEYS = ["neighbourhood", "suburb", "hamlet", "quarter", "city_district"]


def _pick_first(addr: Dict, keys) -> Optional[str]:
    for k in keys:
        v = addr.get(k)
        if v:
            return str(v).strip()
    return None


def reduce_address(addr: Dict) -> Dict:
    """Keep only safe components + boolean flags. No street strings stored."""
    if not isinstance(addr, dict):
        return {"city": None, "region": None, "country": None, "flags": {}}

    flags = {
        "has_too_precise": any(k in addr for k in TOO_PRECISE_KEYS),
        "has_subcity": any(k in addr for k in SUBCITY_KEYS),
        "has_city": any(k in addr for k in CITY_KEYS),
        "has_region": any(k in addr for k in REGION_KEYS),
        "has_country": "country" in addr,
    }

    return {
        "city": _pick_first(addr, CITY_KEYS),
        "region": _pick_first(addr, REGION_KEYS),
        "country": _pick_first(addr, COUNTRY_KEYS),
        "flags": flags,
    }


def classify_precision_safe(reduced: Dict) -> str:
    """
    Classify raw precision from SAFE reduced address.
    Returns: street, sub_city, city, state_province, country, unknown
    """
    if not isinstance(reduced, dict):
        return "unknown"
    flags = reduced.get("flags", {}) or {}

    if flags.get("has_too_precise"):
        return "street"
    if flags.get("has_subcity"):
        return "sub_city"
    if reduced.get("city"):
        return "city"
    if reduced.get("region"):
        return "state_province"
    if reduced.get("country"):
        return "country"
    return "unknown"


def target_precision(reduced: Dict) -> str:
    """
    Our required final precision (max detail we will keep publicly):
    city > state_province > country > unknown
    """
    if not isinstance(reduced, dict):
        return "unknown"
    if reduced.get("city"):
        return "city"
    if reduced.get("region"):
        return "state_province"
    if reduced.get("country"):
        return "country"
    return "unknown"


def build_query(reduced: Dict, precision: str) -> Optional[str]:
    """Build a geocode query for centroid at the desired precision."""
    if not isinstance(reduced, dict):
        return None
    city = reduced.get("city")
    region = reduced.get("region")
    country = reduced.get("country")

    if precision == "city":
        if city and region and country:
            return f"{city}, {region}, {country}"
        if city and country:
            return f"{city}, {country}"
        return None
    if precision == "state_province":
        if region and country:
            return f"{region}, {country}"
        return None
    if precision == "country":
        if country:
            return f"{country}"
        return None
    return None


def round_privacy_coords(lat: float, lon: float, precision: str) -> Tuple[float, float]:
    """
    Make coordinates “look” like the intended precision (extra safety).
    City: 2 decimals (~1km), State: 1 decimal (~11km), Country: 1 decimal
    """
    if precision == "city":
        return (round(lat, 2), round(lon, 2))
    if precision == "state_province":
        return (round(lat, 1), round(lon, 1))
    if precision == "country":
        return (round(lat, 1), round(lon, 1))
    return (lat, lon)


# Nominatim wrappers (slow but robust)
def build_geocoder(timeout_s: int, min_delay_s: float, max_retries: int, error_wait_s: float):
    geolocator = Nominatim(
        user_agent="cites-geoprivacy/1.0 (rakes)",
        timeout=timeout_s
    )

    geocode_rl = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=min_delay_s,
        max_retries=max_retries,
        error_wait_seconds=error_wait_s,
        swallow_exceptions=False
    )

    reverse_rl = RateLimiter(
        geolocator.reverse,
        min_delay_seconds=min_delay_s,
        max_retries=max_retries,
        error_wait_seconds=error_wait_s,
        swallow_exceptions=False
    )

    def safe_geocode(query: str):
        try:
            return geocode_rl(query, exactly_one=True, addressdetails=True, language="en")
        except (GeocoderTimedOut, GeocoderUnavailable, requests.exceptions.RequestException):
            return None

    def safe_reverse(lat: float, lon: float):
        try:
            return reverse_rl((lat, lon), exactly_one=True, addressdetails=True, language="en")
        except (GeocoderTimedOut, GeocoderUnavailable, requests.exceptions.RequestException):
            return None

    return safe_geocode, safe_reverse


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Input CSV path")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--out_xlsx", default=None, help="Optional output XLSX path")
    ap.add_argument("--cache_json", default="CITES_Data/master data/geocode_cache.safe.json", help="Cache JSON path")
    ap.add_argument("--recompute_distance", action="store_true", help="Recompute Distance using privacy coords")
    ap.add_argument("--timeout", type=int, default=15, help="Nominatim timeout seconds")
    ap.add_argument("--min_delay", type=float, default=1.1, help="Min delay between Nominatim calls")
    ap.add_argument("--max_retries", type=int, default=8, help="Max retries in RateLimiter")
    ap.add_argument("--error_wait", type=float, default=3.0, help="Wait on errors before retry")
    ap.add_argument("--cache_flush_every", type=int, default=50, help="Write cache every N reverse results")
    args = ap.parse_args()

    # Read without auto-converting "NA" strings into NaN
    df = pd.read_csv(args.in_csv, dtype=str, encoding="utf-8-sig", keep_default_na=False)

    # Numeric columns (if present)
    for col in ["Latitude", "Longitude", "COPlatitude", "COPlongitude", "Distance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace({"": np.nan, "NA": np.nan}), errors="coerce")

    # Add required new output columns
    for col in ["geo_precision", "Latitude_privacy", "Longitude_privacy", "geo_too_precise_raw"]:
        if col not in df.columns:
            df[col] = "NA"

    # If coords not present, we can still output the column
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        print("WARNING: No Latitude/Longitude columns found. Writing file with geo_precision=NA only.")
        df.to_csv(args.out_csv, index=False, encoding="utf-8-sig", na_rep="NA")
        if args.out_xlsx:
            df.to_excel(args.out_xlsx, index=False, na_rep="NA")
        print("DONE")
        return

    safe_geocode, safe_reverse = build_geocoder(
        timeout_s=args.timeout,
        min_delay_s=args.min_delay,
        max_retries=args.max_retries,
        error_wait_s=args.error_wait,
    )

    cache = load_cache(args.cache_json)

    # Unique coordinate pairs
    coord_df = df[["Latitude", "Longitude"]].dropna().copy()
    coord_df["lat_r"] = coord_df["Latitude"].round(5)
    coord_df["lon_r"] = coord_df["Longitude"].round(5)
    unique_coords = coord_df[["lat_r", "lon_r"]].drop_duplicates().values.tolist()

    print(f"Unique coordinate pairs to verify: {len(unique_coords)}")

    # Mapping: (lat_r, lon_r) -> (geo_precision_final, too_precise_raw, lat_priv, lon_priv)
    results: Dict[Tuple[float, float], Tuple[str, str, float, float]] = {}

    flush_counter = 0

    for lat_r, lon_r in tqdm(unique_coords, desc="Reverse-geocoding unique coords"):
        key = f"{lat_r},{lon_r}"

        # Reverse cache (SAFE only)
        reduced = cache["reverse_safe"].get(key)

        if reduced is None:
            loc = safe_reverse(float(lat_r), float(lon_r))
            if loc is None:
                reduced = {"city": None, "region": None, "country": None, "flags": {}}
            else:
                addr = getattr(loc, "raw", {}).get("address", None)
                reduced = reduce_address(addr or {})

            cache["reverse_safe"][key] = reduced
            flush_counter += 1

            if flush_counter >= args.cache_flush_every:
                save_cache(args.cache_json, cache)
                flush_counter = 0

        raw_prec = classify_precision_safe(reduced)
        too_precise = "1" if raw_prec in {"street", "sub_city"} else "0"

        final_prec = target_precision(reduced)  # city/state_province/country/unknown
        query = build_query(reduced, final_prec)

        # Default privacy coords: keep original if we cannot safely coarsen
        lat_priv, lon_priv = float(lat_r), float(lon_r)

        # If we have a query, geocode to centroid of that admin unit
        if query:
            geo = cache["geocode_safe"].get(query)
            if geo is None:
                g = safe_geocode(query)
                if g is None:
                    geo = {"lat": None, "lon": None}
                else:
                    geo = {"lat": float(g.latitude), "lon": float(g.longitude)}
                cache["geocode_safe"][query] = geo
                flush_counter += 1
                if flush_counter >= args.cache_flush_every:
                    save_cache(args.cache_json, cache)
                    flush_counter = 0

            if geo and geo.get("lat") is not None and geo.get("lon") is not None:
                lat_priv, lon_priv = float(geo["lat"]), float(geo["lon"])
                lat_priv, lon_priv = round_privacy_coords(lat_priv, lon_priv, final_prec)

        # If final_prec unknown, don’t claim precision; privacy coords become NA
        if final_prec == "unknown":
            results[(lat_r, lon_r)] = ("unknown", too_precise, np.nan, np.nan)
        else:
            results[(lat_r, lon_r)] = (final_prec, too_precise, lat_priv, lon_priv)

    # Final cache flush
    save_cache(args.cache_json, cache)

    # Apply back to df
    df["lat_r"] = df["Latitude"].round(5)
    df["lon_r"] = df["Longitude"].round(5)

    def apply_row(r):
        if pd.isna(r["lat_r"]) or pd.isna(r["lon_r"]):
            return pd.Series(["NA", "NA", np.nan, np.nan])
        final_prec, too_precise, latp, lonp = results.get(
            (r["lat_r"], r["lon_r"]),
            ("unknown", "0", np.nan, np.nan)
        )
        return pd.Series([final_prec, too_precise, latp, lonp])

    df[["geo_precision", "geo_too_precise_raw", "Latitude_privacy", "Longitude_privacy"]] = df.apply(apply_row, axis=1)

    # Optional recompute Distance using privacy coords
    if args.recompute_distance:
        if "COPlatitude" in df.columns and "COPlongitude" in df.columns:
            df["Distance"] = df.apply(
                lambda rr: haversine_km(rr.get("COPlatitude"), rr.get("COPlongitude"), rr.get("Latitude_privacy"), rr.get("Longitude_privacy")),
                axis=1
            )
        else:
            print("WARNING: COPlatitude/COPlongitude not found; cannot recompute Distance.")

    # Write outputs
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig", na_rep="NA")
    if args.out_xlsx:
        df.to_excel(args.out_xlsx, index=False, na_rep="NA")

    print("\ngeo_precision counts:")
    print(df["geo_precision"].value_counts(dropna=False).head(20))

    print("\nDONE")
    print(f"Saved CSV : {args.out_csv}")
    if args.out_xlsx:
        print(f"Saved XLSX: {args.out_xlsx}")
    print(f"Cache used: {args.cache_json}")


if __name__ == "__main__":
    main()
