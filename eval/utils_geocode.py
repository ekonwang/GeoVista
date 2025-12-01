import json
import re
import os
import unicodedata
from typing import Any, Callable, Dict, Optional, Tuple

from utils_api import chat_gpt5_nano as chat_fn

# =========================
# Common helpers
# =========================

def _ascii_lower(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    # Normalize all commas/whitespace to ', ' separators
    s = re.sub(r"[,\s]+", " ", s)
    s = re.sub(r"\s*,\s*", ", ", s.replace(",", " , "))
    s = re.sub(r"\s+", " ", s).strip(" ,")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return ", ".join(parts) if parts else s

def _extract_json_obj(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s = s.strip()
    # Direct JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fallback: use the first {...} segment
    start = s.find("{"); end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = s[start:end+1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None

# =========================
# Function 1: extract address using 4o-mini (predictive address extractor)
# ========================

def extract_pred_address_v2(
    text: str,
    # chat_fn: Callable[[list, Optional[str], int], str],
    api_key: Optional[str] = None,
    timeout: int = 30,
    debug: bool = False,
    normalize: bool = False,
) -> str:
    """
    Extract the final address string from free text (pred address). Always returns a str.
    - chat_fn: e.g., chat_gpt5_nano(messages, api_key=None, timeout=30) -> str
    - On failure or when unparsable, return the fixed fallback: "honiara, guadalcanal, solomon islands"
    - When debug=True, prints each attempt's strictness level, raw response, and parsing errors.
    """
    FALLBACK_ADDR = "honiara, guadalcanal, solomon islands"

    def _messages(payload: Dict[str, Any], strict_level: int) -> list:
        sys = (
            "You are a precise address extractor for geocoding. "
            "You will receive a JSON payload from the user; the field 'text' contains free-form reasoning that often ends with a FINAL explicit address. "
            "Return JSON ONLY with a single key: address (string). "
            "Extract the single most explicit and complete FINAL address mentioned. Prefer lines such as 'Therefore, the location ... is:', 'The location ... is:', 'Final address:', 'Location:'. "
            "If both a POI/building name and the street address are given in that final line, include the POI first, then the full address, comma-separated. "
            "Include street number and street name when present; include city, state/province (use the 2-letter US code if present), postal/ZIP code, and country when present. "
            "Do not invent or alter components; keep abbreviations exactly as written (e.g., 'Ave', 'Rd', 'St', 'FL', 'USA'). "
            "Normalize formatting: use comma followed by a single space between parts, collapse multiple spaces, remove surrounding quotes and any trailing punctuation (.,;:). "
            "Use official English names when multiple variants appear."
        )
        # Add a concrete teaching example to bias extraction toward the FINAL address line
        example_in = (
            "4.  **Confirmation:** Searching for this location on Google Maps leads to the **Young Transportation Center at 3915 Michigan Avenue, Fort Myers, Florida**. "
            "By examining the Google Street View imagery at this address, I can confirm that the building, the sign, the flagpole, the parking lot layout, and the surrounding environment are an exact match to the provided panorama.\n\n"
            "Therefore, the location of the image is:\n\n"
            "**Young Transportation Center, 3915 Michigan Ave, Fort Myers, FL 33916, USA.**\n\n"
            "The specific image is a Google Maps photosphere taken from the parking lot of the facility."
        )
        example_out = {"address": "Young Transportation Center, 3915 Michigan Ave, Fort Myers, FL 33916, USA"}

        sys_examples = (
            "\n\nExample Input.text => " + example_in +
            "\nExample Output => " + json.dumps(example_out)
        )

        if strict_level >= 1:
            sys += (
                " Your previous output was invalid. Return EXACTLY {\"address\": \"...\"} with no extra text. "
                "Only the JSON object, no code fences."
            )
        if strict_level >= 2:
            sys += (
                " Only output the JSON object. Do not explain, do not add any additional fields or text."
            )

        return [
            {"role": "system", "content": sys + sys_examples},
            {"role": "user", "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]},
        ]

    payload = {
        "task": "extract_final_address",
        "text": text,
        # Emphasize full, final address extraction (including POI name if present)
        "format_hint": "POI/building, street number and street, city, state/province code, postal code, country (comma-separated, English)",
    }

    last_err = None
    for attempt in range(3):
        try:
            msgs = _messages(payload, strict_level=attempt)
            if debug:
                print(f"[extract_pred_address] attempt={attempt} sending messages")
            resp = chat_fn(msgs, api_key=api_key, timeout=timeout)
            if debug:
                preview = (resp[:300] + "...") if isinstance(resp, str) and len(resp) > 300 else resp
                print(f"[extract_pred_address] raw response: {preview!r}")
            obj = _extract_json_obj(resp)
            if not isinstance(obj, dict):
                last_err = "json_obj_none"
                if debug:
                    print(f"[extract_pred_address] parse failed: {last_err}")
                continue
            addr = obj.get("address")
            if isinstance(addr, str):
                if normalize:
                    addr_norm = _ascii_lower(addr)
                else:
                    addr_norm = addr
                if addr_norm:
                    if debug:
                        print(f"[extract_pred_address] parsed address: {addr_norm}")
                    return addr_norm
            last_err = "address_field_invalid"
            if debug:
                print(f"[extract_pred_address] parse failed: {last_err}, obj={obj}")
        except Exception as e:
            last_err = f"exception:{e}"
            if debug:
                print(f"[extract_pred_address] exception: {last_err}")

    if debug:
        print(f"[extract_pred_address] fallback due to: {last_err}")
    return FALLBACK_ADDR


# =========================
# Function 2: geocode address (predictive address -> geolocation point)
# =========================

def geocode_address(
    address: str,
    google_api_key: Optional[str] = None,
    timeout: int = 20,
    debug: bool = False,
    allow_fallback: bool = True,
) -> Dict[str, float]:
    """
    Resolve the address to latitude/longitude.
    - Prefer Google Geocoding API (when google_api_key is provided), and fall back to OSM Nominatim on failure.
    - On success returns: {"lat": <float>, "lng": <float>}
    - On failure raises ValueError.
    - When debug=True, prints which service/URL (key redacted), response snippets, and errors.
    """
    import requests
    from urllib.parse import urlencode

    # default fallback for "honiara, guadalcanal, solomon islands"
    fall_back = {'lat': -9.4306698, 'lng': 159.9526758}
    google_api_key = os.getenv("GOOGLE_MAPS_API_KEY", None)

    address = (address or "").strip()
    if not address or 'unknown' in address.lower():
        if allow_fallback:
            return fall_back
        raise ValueError("Empty address for geocoding")

    # --- 1) Google Geocoding API ---
    if google_api_key:
        try:
            params = {"address": address, "key": google_api_key}
            url = f"https://maps.googleapis.com/maps/api/geocode/json?{urlencode(params)}"
            url_dbg = url.replace(google_api_key, "****")  # Avoid leaking the key
            if debug:
                print(f"[geocode] trying Google Geocoding: {url_dbg}")
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if debug:
                print(f"[geocode] Google status={data.get('status')}, results={len(data.get('results', []))}")
            if data.get("status") in ("OK",) and data.get("results"):
                loc = data["results"][0]["geometry"]["location"]
                lat = float(loc["lat"]); lng = float(loc["lng"])
                if debug:
                    print(f"[geocode] Google success lat={lat}, lng={lng}")
                return {"lat": lat, "lng": lng}
            else:
                if debug:
                    print(f"[geocode] Google failed payload snippet={str(data)[:300]}")
        except Exception as e:
            if debug:
                print(f"[geocode] Google exception: {e}")
    else:
        raise ValueError("Expect `GOOGLE_MAPS_API_KEY`!!")

    # --- 2) OSM Nominatim fallback ---
    try:
        params = {"q": address, "format": "json", "limit": 1}
        headers = {"User-Agent": "geo-eval/1.0 (contact: you@example.com)"}
        if debug:
            print(f"[geocode] trying OSM Nominatim with params={params}")
        r = requests.get("https://nominatim.openstreetmap.org/search", params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        arr = r.json()
        if debug:
            print(f"[geocode] Nominatim results={len(arr) if isinstance(arr, list) else 'N/A'}")
        if isinstance(arr, list) and arr:
            lat = float(arr[0]["lat"]); lng = float(arr[0]["lon"])
            if debug:
                print(f"[geocode] Nominatim success lat={lat}, lng={lng}")
            return {"lat": lat, "lng": lng}
        else:
            if debug:
                print(f"[geocode] Nominatim empty response snippet={str(arr)[:300]}")
    except Exception as e:
        if debug:
            print(f"[geocode] Nominatim exception: {e}")

    if allow_fallback:
        print(f"[geocode] using fallback for {address}, location = {fall_back}")
        return fall_back
    raise ValueError(f"Failed to geocode address: {address}")


# =========================
# Function 3: distance between two points (km)
# =========================

def _coerce_lat_lng(d: Dict[str, Any]) -> Tuple[float, float]:
    if not isinstance(d, dict):
        raise ValueError("Point must be a dict with lat/lng")
    cand_lat = ["lat", "latitude"]
    cand_lng = ["lng", "lon", "long", "longitude"]
    lat = None; lng = None
    for k in cand_lat:
        if k in d and d[k] is not None:
            lat = float(d[k]); break
    for k in cand_lng:
        if k in d and d[k] is not None:
            lng = float(d[k]); break
    if lat is None or lng is None:
        raise ValueError(f"Invalid point dict (need lat/lng): {d}")
    return lat, lng

def haversine_km(p1: Dict[str, Any], p2: Dict[str, Any]) -> float:
    """
    Compute great-circle distance between two points (kilometers).
    p1 / p2: dict containing lat / lng (or latitude / longitude / lon / long)
    Uses WGS84 mean Earth radius 6371.0088 km
    """
    import math
    lat1, lng1 = _coerce_lat_lng(p1)
    lat2, lng2 = _coerce_lat_lng(p2)

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    R = 6371.0088  # km
    return R * c


if __name__ == "__main__":
    TEST_MESSAGE = "Based on the visual clues in the image, I have located the place.\n\n**Analysis of Clues:**\n\n1.  **Signage:** In the cropped images, several business signs are visible. One clearly reads \"**STEILO BAR**\". Another sign next to it is for \"**tipwin**,\" a sports betting company. A sign above the bar indicates a \"**FRISEUR**\" (German for hairdresser).\n2.  **License Plates:** A German license plate is visible on a black car parked on the left. The city code appears to be \"**HH**,\" which stands for Hansestadt Hamburg (Hanseatic City of Hamburg).\n3.  **Architecture:** The buildings, particularly the large apartment blocks and the mixed-use commercial building with a parking garage, are characteristic of post-war urban development in many German cities.\n\n**Search and Verification:**\n\nCombining these clues, a search for \"STEILO BAR Hamburg\" immediately leads to an address. Verifying this location on Google Maps confirms the presence of both the Steilo Bar and a tipwin betting shop at the **Einkaufszentrum (shopping center) Rahlstedt-Ost**.\n\nThe photo was taken on **Schöneberger Straße** in Hamburg, Germany, looking towards the shopping center.\n\n**Conclusion:**\n\nThe location of the image is on **Schöneberger Straße, 22149 Hamburg, Germany**, in front of the Einkaufszentrum Rahlstedt-Ost.\n\nYou can see this exact spot on [Google Maps Street View](https://www.google.com/maps/@53.5999251,10.1557007,3a,75y,283.43h,94.94t/data=!3m6!1e1!3m4!1s0Qv7M-Wc1tI-L0R-Gf0YfA!2e0!7i16384!8i8192?entry=ttu)."


    text = TEST_MESSAGE
    addr = extract_pred_address_v2(text, timeout=30, debug=True)
    addr_now = "Tencent beijing Office, Beijing, China"

    coord_extract = geocode_address(addr, timeout=20, debug=True)
    coord_now = geocode_address(addr_now, timeout=20, debug=True)
    print(f"The location of {addr} is {coord_extract}")
    print(f"The location of {addr_now} is {coord_now}")

    dist = haversine_km(coord_extract, coord_now)
    print('The distance between the two locations is ' + str(dist) + ' km')
