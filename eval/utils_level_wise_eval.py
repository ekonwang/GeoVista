# eval/multi_level_eval_gemini_0825.py
import json
import re
import unicodedata
from typing import Dict, Optional, Tuple, Any

# from utils_gpt import chat_4o_mini
# from utils_api import chat_4o_mini, chat_gemini
from utils_api import chat_gpt5_nano as chat_fn
from utils import print_hl


def _normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _contains_whole_term(text_norm: str, term_norm: str) -> bool:
    if not term_norm:
        return False
    pattern = r"(?<!\w)" + re.escape(term_norm) + r"(?!\w)"
    return re.search(pattern, text_norm) is not None


def _extract_json_obj(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s = s.strip()
    # Try direct parse first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fallback: extract first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = s[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _read_possible_lat_lng(d: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if not isinstance(d, dict):
        return None
    cand_lat_keys = ["lat", "latitude"]
    cand_lng_keys = ["lng", "lon", "long", "longitude"]
    lat = None
    lng = None
    for k in cand_lat_keys:
        if k in d and d[k] is not None:
            lat = d[k]
            break
    for k in cand_lng_keys:
        if k in d and d[k] is not None:
            lng = d[k]
            break
    if lat is None or lng is None:
        return None
    try:
        return float(lat), float(lng)
    except Exception:
        return None


def _read_possible_text_fields(d: Dict[str, Any]) -> Dict[str, Optional[str]]:
    if not isinstance(d, dict):
        return {"country": None, "province_or_state": None, "city": None}

    country_keys = ["country", "country_name", "nation"]
    state_keys = [
        "province_or_state",
        "state",
        "province",
        "state_province",
        "admin1",
        "region",
        "state_name",
        "province_name",
    ]
    city_keys = ["city", "city_name", "town", "locality", "admin2", "county"]

    def pick(keys):
        for k in keys:
            if k in d and d[k]:
                return str(d[k])
        return None

    return {
        "country": pick(country_keys),
        "province_or_state": pick(state_keys),
        "city": pick(city_keys),
    }


def eval_geolocation_response(
    response: str,
    loc_dict: Dict[str, Optional[str]],
    model_verifier: bool = False,
    api_key: Optional[str] = None,
    timeout: int = 120,
    debug_mode: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate whether a model's geolocation answer matches loc_dict.
    Flow:
    1) First run local rule-based matching to get rb = {country_correct, state_correct, city_correct}.
    2) If model_verifier=True, call Gemini once to get mb (same three booleans);
       then merge with rb using bitwise OR: combined = rb OR mb.
    3) Apply hierarchical consistency to combined:
       - If city_correct=True, also set state_correct=True and country_correct=True.
       - Else if state_correct=True, set country_correct=True.
    4) Return the adjusted combined.
    """

    # ------ Rule-based matching (local) ------
    def rule_based() -> Dict[str, bool]:
        res_norm = _normalize_text(response or "")
        city_norm = _normalize_text(loc_dict.get("city"))
        state_norm = _normalize_text(
            loc_dict.get("province_or_state") or loc_dict.get("state")
        )
        country_norm = _normalize_text(loc_dict.get("country"))

        city_hit = _contains_whole_term(res_norm, city_norm) if city_norm else False
        state_hit = _contains_whole_term(res_norm, state_norm) if state_norm else False
        country_hit = _contains_whole_term(res_norm, country_norm) if country_norm else False

        city_correct = city_hit
        state_correct = state_hit or city_hit
        country_correct = country_hit or state_hit or city_hit

        return {
            "country_correct": bool(country_correct),
            "state_correct": bool(state_correct),
            "city_correct": bool(city_correct),
        }

    rb = rule_based()  # 1) Always do local rule-based matching first
    print_hl("[eval_geolocation_response] rule based raw response:")
    print(rb if isinstance(rb, str) else json.dumps(rb, ensure_ascii=False))

    # If not using model verifier, apply hierarchical consistency and return
    if not model_verifier:
        combined = dict(rb)
        # 3) Hierarchical consistency fix
        if combined["city_correct"]:
            combined["state_correct"] = True
            combined["country_correct"] = True
        elif combined["state_correct"]:
            combined["country_correct"] = True
        return combined

    # ------ Model verifier (single call) ------
    def _model_verdict() -> Optional[Dict[str, bool]]:
        try:
            sys_prompt = (
                "You are a strict evaluator. Decide if a free-text geolocation answer matches a gold location.\n"
                "Rules:\n"
                "1) If the answer names the correct city (as a toponym), then city/state/country are all True.\n"
                "2) If it names the correct state/province (but not the correct city), then state and country are True; city is False.\n"
                "3) If it names only the correct country, then only country is True.\n"
                "4) If none match, all are False.\n"
                "5) Consider common synonyms and English exonyms; ignore punctuation and case.\n"
                "Respond with a single JSON object: {\"country_correct\": <bool>, \"state_correct\": <bool>, \"city_correct\": <bool>}."
            )
            user_payload = {
                "gold_location": {
                    "country": loc_dict.get("country"),
                    "province_or_state": loc_dict.get("province_or_state") or loc_dict.get("state"),
                    "city": loc_dict.get("city"),
                },
                "model_response": response,
            }
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}]},
            ]
            resp = chat_fn(messages, timeout=timeout)
            if debug_mode and model_verifier:
                try:
                    print_hl("[eval_geolocation_response] model_verifier raw response:")
                    print(resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False))
                except Exception:
                    pass
            obj = _extract_json_obj(resp)
            if isinstance(obj, dict):
                return {
                    "country_correct": bool(obj.get("country_correct") is True),
                    "state_correct": bool(obj.get("state_correct") is True),
                    "city_correct": bool(obj.get("city_correct") is True),
                }
            else:
                raise ValueError(f'Could not extract json object from raw response: {resp}')
        except Exception as e:
            raise e

    mb = _model_verdict()

    # 2) Use model verifier result
    combined = mb

    # 3) Hierarchical consistency fix
    if combined["city_correct"]:
        combined["state_correct"] = True
        combined["country_correct"] = True
    elif combined["state_correct"]:
        combined["country_correct"] = True

    # 4) Return
    return combined
