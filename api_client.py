import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

try:
    import streamlit as st
except ImportError:  # plain scripts and the GitHub build run without Streamlit
    st = None

_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_app_id() -> str | None:
    """
    The Wargaming application id: the WG_APPLICATION_ID environment variable,
    else the untracked file .wg_application_id next to this module. It is kept
    out of the source so this file can be copied to a public repository.
    """
    value = os.environ.get("WG_APPLICATION_ID", "").strip()
    if value:
        return value
    path = os.path.join(_HERE, ".wg_application_id")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            value = f.read().strip()
        if value:
            return value
    return None


def _cache_data(**kwargs):
    """st.cache_data when Streamlit is installed, otherwise no caching."""
    return st.cache_data(**kwargs) if st else (lambda f: f)


def _ui_error(msg: str) -> None:
    """st.error inside the Streamlit app, stderr otherwise."""
    if st:
        st.error(msg)
    else:
        print(msg, file=sys.stderr)


API_KEY = _resolve_app_id()
DEFAULT_ACCOUNT_ID = 1009796686

# Hardcoded quick-load clan, mirroring DEFAULT_ACCOUNT_ID. Clan ids are
# per-region, so this pairs with the region it was looked up on.
DEFAULT_CLAN_ID     = 4455
DEFAULT_CLAN_LABEL  = "[_V_] Void"
DEFAULT_CLAN_REGION = "NA"
TIMEOUT = 20
RETRIES = 3
BACKOFF = 1.5
# API errors that mean "try again shortly". The page's live panel and the
# research scans can share one application id, so a busy moment is expected.
RETRYABLE_API_ERRORS = ("REQUEST_LIMIT_EXCEEDED", "SOURCE_NOT_AVAILABLE")
API_ERROR_WAITS = (2, 4, 8, 16, 30)   # seconds before each retry, about a minute in all

# Batch tuning. account/info/ accepts up to 100 comma-separated account_ids in
# one request. tanks/stats/ does NOT batch — more than one id returns
# INVALID_ACCOUNT_ID — so it stays one account per request and we win back the
# time by running those requests across a thread pool instead.
INFO_CHUNK    = 100
TANKS_CHUNK   = 1
MAX_WORKERS   = 8
TANKS_WORKERS = 12

REGION_URLS = {
    "NA":   "https://api.wotblitz.com/wotb/",
    "EU":   "https://api.wotblitz.eu/wotb/",
    "APAC": "https://api.wotblitz.asia/wotb/",
}
DEFAULT_REGION = "NA"


def _get(endpoint: str, params: dict, base_url: str = REGION_URLS["NA"],
         quiet: bool = False) -> dict | None:
    """
    GET an API endpoint with retries.

    `quiet=True` suppresses st.* calls — required when running inside a worker
    thread, where there is no Streamlit script context.
    """
    if API_KEY is None:
        raise RuntimeError(
            "Wargaming application id not set: set WG_APPLICATION_ID or put it "
            "in .wg_application_id next to api_client.py")
    params["application_id"] = API_KEY
    last_exc = None
    failures = 0                       # timeouts and connection errors
    api_waits = list(API_ERROR_WAITS)  # retries left after a retryable API error
    while True:
        try:
            resp = requests.get(base_url + endpoint, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "error":
                message = (data.get("error") or {}).get("message", "Unknown error")
                if message in RETRYABLE_API_ERRORS and api_waits:
                    time.sleep(api_waits.pop(0))
                    continue
                if not quiet:
                    _ui_error(f"API error: {message}")
                return None
            return data
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            failures += 1
            if failures >= RETRIES:
                break
            time.sleep(BACKOFF * failures)
        except requests.RequestException as e:
            if not quiet:
                _ui_error(f"Request failed: {e}")
            return None
    if not quiet:
        _ui_error(f"Request failed after {RETRIES} attempts: {last_exc}")
    return None


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


@_cache_data(ttl=300)
def search_player(name: str, base_url: str = REGION_URLS["NA"]) -> list:
    """Search for a player by name (exact match). Returns list of {account_id, nickname}."""
    data = _get("account/list/", {"search": name, "type": "exact"}, base_url)
    if data is None:
        return []
    return data.get("data", [])


@_cache_data(ttl=300)
def get_account_info(account_id: int, base_url: str = REGION_URLS["NA"]) -> dict:
    """Fetch account summary. Returns dict with nickname, battles, wins, etc."""
    data = _get("account/info/", {"account_id": account_id}, base_url)
    if data is None:
        return {}
    accounts = data.get("data", {})
    return accounts.get(str(account_id), {}) or {}


@_cache_data(ttl=300)
def search_clan(tag: str, base_url: str = REGION_URLS["NA"]) -> list:
    """Search clans by tag or name. Returns list of {clan_id, tag, name, members_count}."""
    data = _get("clans/list/", {"search": tag, "limit": 10}, base_url)
    if data is None:
        return []
    return data.get("data", [])


@_cache_data(ttl=300)
def get_clan_info(clan_id: int, base_url: str = REGION_URLS["NA"]) -> dict:
    """Fetch clan info including member list. Returns clan dict."""
    data = _get("clans/info/", {"clan_id": clan_id, "extra": "members"}, base_url)
    if data is None:
        return {}
    return data.get("data", {}).get(str(clan_id), {}) or {}


TANK_STATS_FIELDS = ",".join([
    "tank_id",
    "all.battles",
    "all.wins",
    "all.damage_dealt",
    "all.frags",
    "all.survived_battles",
    "all.spotted",
    "all.dropped_capture_points",
])


@_cache_data(ttl=300)
def get_tank_stats(account_id: int, base_url: str = REGION_URLS["NA"]) -> list:
    """
    Fetch per-tank stats for a player.
    Returns list of {tank_id, all: {battles, wins, damage_dealt, frags, spotted, dropped_capture_points}}
    """
    data = _get("tanks/stats/", {"account_id": account_id, "fields": TANK_STATS_FIELDS}, base_url)
    if data is None:
        return []
    tank_list = data.get("data", {}).get(str(account_id))
    if not tank_list:
        return []
    return tank_list


# ── Batched / parallel fetches (used by the Clan tab) ─────────────────────────

def _fetch_info_chunk(ids: list, base_url: str) -> dict:
    """One account/info/ request for up to INFO_CHUNK ids. Returns {int_id: info}."""
    data = _get(
        "account/info/",
        {"account_id": ",".join(str(i) for i in ids)},
        base_url,
        quiet=True,
    )
    if data is None:
        return {}
    out = {}
    for k, v in (data.get("data") or {}).items():
        if v:
            out[int(k)] = v
    return out


def _fetch_tanks_chunk(ids: list, base_url: str) -> dict:
    """
    One tanks/stats/ request. The endpoint takes a single account_id only, so
    `ids` is always length 1 — parallelism, not batching, is what speeds this up.
    Returns {int_id: [tanks]}.
    """
    data = _get(
        "tanks/stats/",
        {"account_id": ids[0], "fields": TANK_STATS_FIELDS},
        base_url,
        quiet=True,
    )
    if data is None:
        return {}
    out = {}
    for k, v in (data.get("data") or {}).items():
        out[int(k)] = v or []
    return out


def _parallel_chunks(account_ids, chunk_size, worker, base_url, progress_cb=None,
                     max_workers: int = MAX_WORKERS) -> dict:
    """
    Split account_ids into chunks, fetch them across a thread pool, and merge.
    progress_cb(done, total) is invoked on the calling thread as chunks land.
    """
    ids = list(account_ids)
    if not ids:
        return {}

    chunks = list(_chunks(ids, chunk_size))
    merged, done = {}, 0

    with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as pool:
        futures = {pool.submit(worker, c, base_url): c for c in chunks}
        for fut in as_completed(futures):
            try:
                merged.update(fut.result())
            except Exception:
                pass  # a failed chunk leaves its ids absent; caller treats as no data
            done += len(futures[fut])
            if progress_cb:
                progress_cb(min(done, len(ids)), len(ids))

    return merged


def get_accounts_info_batch(account_ids, base_url: str = REGION_URLS["NA"],
                            progress_cb=None) -> dict:
    """Fetch account/info/ for many players at once. Returns {account_id: info}."""
    return _parallel_chunks(account_ids, INFO_CHUNK, _fetch_info_chunk, base_url, progress_cb)


def get_tank_stats_batch(account_ids, base_url: str = REGION_URLS["NA"],
                         progress_cb=None) -> dict:
    """Fetch tanks/stats/ for many players at once. Returns {account_id: [tanks]}."""
    return _parallel_chunks(account_ids, TANKS_CHUNK, _fetch_tanks_chunk, base_url,
                            progress_cb, max_workers=TANKS_WORKERS)
