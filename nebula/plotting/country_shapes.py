from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Literal

import cartopy
from cartopy.io.shapereader import Reader, natural_earth

from nebula_old import NEBULA_ROOT_DIR

# -----------------------------------------------------------------------------
# Cartopy data directory setup
# -----------------------------------------------------------------------------

project_root = os.path.dirname(NEBULA_ROOT_DIR)
cartopy.config["pre_existing_data_dir"] = os.path.join(project_root, "data", "cartopy")
cartopy.config["data_dir"] = os.path.join(project_root, "data", "cartopy")

# ----------------------------
# Utilities
# ----------------------------


def _norm_text(s: str) -> str:
    """
    Normalize text for fuzzy matching:
      - casefold
      - strip accents
      - collapse non-alnum to single spaces
    """
    s = "" if s is None else str(s)
    s = s.casefold()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9a-z]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _token_set(s: str) -> set[str]:
    s = _norm_text(s)
    return set(t for t in s.split(" ") if t)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _best_window_substring_bonus(query_norm: str, cand_norm: str) -> float:
    """
    Small bonus if query appears as a substring in candidate, or vice versa.
    Returns in [0, 1].
    """
    if not query_norm or not cand_norm:
        return 0.0
    if query_norm in cand_norm:
        # longer matches should get slightly higher bonus
        return min(1.0, len(query_norm) / max(1.0, len(cand_norm)))
    if cand_norm in query_norm:
        return min(1.0, len(cand_norm) / max(1.0, len(query_norm)))
    return 0.0


# ----------------------------
# Match result
# ----------------------------


@dataclass(frozen=True)
class CountryMatch:
    score: float
    record: object  # cartopy.io.shapereader.Record
    key: str  # which attribute key matched best
    value: str  # matched attribute value (raw)
    display_name: str  # best-effort name for display (ADMIN/NAME)
    iso_a3: Optional[str]
    iso_a2: Optional[str]


# ----------------------------
# Core fuzzy matcher
# ----------------------------


def fuzzy_find_countries(
    query: str,
    resolution: Literal["10m", "50m", "110m"] = "110m",
    *,
    topk: int = 10,
    # Attribute keys to search (in priority order). You can extend/reorder.
    keys: Sequence[str] = (
        "ADM0_A3",
        "ISO_A3",
        "ISO_A2",
        "POSTAL",
        "ABBREV",
        "NAME",
        "NAME_LONG",
        "ADMIN",
        "FORMAL_EN",
        "SOVEREIGNT",
        "GEOUNIT",
        "SUBUNIT",
        "BRK_NAME",
        "NAME_EN",
        "NAME_ES",
        "NAME_FR",
        "NAME_DE",
        "NAME_PT",
        "NAME_RU",
        "NAME_AR",
        "NAME_ZH",
        "NAME_JA",
        "NAME_KO",
    ),
    min_score: float = 0.35,
) -> List[CountryMatch]:
    """
    Fuzzy-match Natural Earth admin_0_countries records to a query string.

    Scoring strategy (robust without extra deps):
      1) Exact match on code fields (ADM0_A3/ISO_A3/ISO_A2/POSTAL) -> score ~ 1.0
      2) Token-set Jaccard similarity on name fields -> [0,1]
      3) Substring bonus (query in value or value in query)

    Returns topk matches sorted by descending score.

    Usage:
        reader = Reader(natural_earth(...))
        matches = fuzzy_find_countries(reader, "United States", topk=5)
        best = matches[0].record
    """
    q = _norm_text(query)
    q_tokens = _token_set(query)

    # Common code-like queries: strip spaces/punct for code comparison
    q_code = re.sub(r"[^0-9a-z]+", "", q)

    code_keys = {"ADM0_A3", "ISO_A3", "ISO_A2", "POSTAL", "ADM0_ISO", "WB_A2", "WB_A3"}

    results: List[CountryMatch] = []
    reader = Reader(natural_earth(resolution, "cultural", "admin_0_countries"))

    for rec in reader.records():
        attrs = rec.attributes

        # basic identity fields for display
        display_name = (
            attrs.get("ADMIN") or attrs.get("NAME") or attrs.get("NAME_LONG") or ""
        )
        iso_a3 = attrs.get("ISO_A3") or attrs.get("ADM0_A3") or None
        iso_a2 = attrs.get("ISO_A2") or attrs.get("POSTAL") or None

        best_score = 0.0
        best_key = ""
        best_val = ""

        for k in keys:
            v = attrs.get(k)
            if not v:
                continue

            v_str = str(v)
            v_norm = _norm_text(v_str)

            # Code exact/near-exact handling
            if k in code_keys:
                v_code = re.sub(r"[^0-9a-z]+", "", v_norm)
                if v_code and q_code:
                    if v_code == q_code:
                        s = 1.0
                    elif v_code.startswith(q_code) or q_code.startswith(v_code):
                        s = 0.92
                    else:
                        s = 0.0
                else:
                    s = 0.0
            else:
                # Name-like handling: token similarity + substring bonus
                v_tokens = _token_set(v_str)
                s = _jaccard(q_tokens, v_tokens)
                s = max(s, _best_window_substring_bonus(q, v_norm))

                # Mild preference for shorter "clean" matches (e.g., "United States")
                if s > 0:
                    length_penalty = 1.0 / (1.0 + 0.015 * max(0, len(v_norm) - len(q)))
                    s *= length_penalty

            if s > best_score:
                best_score = s
                best_key = k
                best_val = v_str

            # short-circuit if perfect match found
            if best_score >= 1.0:
                break

        if best_score >= min_score:
            results.append(
                CountryMatch(
                    score=float(best_score),
                    record=rec,
                    key=best_key,
                    value=best_val,
                    display_name=str(display_name),
                    iso_a3=str(iso_a3) if iso_a3 is not None else None,
                    iso_a2=str(iso_a2) if iso_a2 is not None else None,
                )
            )

    results.sort(key=lambda m: m.score, reverse=True)
    return results[: max(1, int(topk))]


def fuzzy_find_country_record(
    query: str,
    resolution: Literal["10m", "50m", "110m"] = "110m",
    *,
    topk: int = 5,
    min_score: float = 0.35,
) -> Tuple[object, List[CountryMatch]]:
    """
    Convenience wrapper: returns (best_record, matches).
    Raises ValueError if no matches above threshold.
    """
    matches = fuzzy_find_countries(query, resolution, topk=topk, min_score=min_score)
    if not matches:
        raise ValueError(
            f"No country matches found for {query!r} (min_score={min_score})"
        )
    return matches[0].record, matches
