"""Generic disk-cache and retrying-HTTP-GET helpers shared by every module
that talks to a remote REST API (Metabolomics Workbench, in this package)."""

import json
import os
import time
from pathlib import Path

import requests


def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)  # atomic write, avoids corruption on crash


def get_json(url, retries=3, backoff=2.0, timeout=30, no_retry_statuses=()):
    """requests.get(url).json() with a few retries -- a full-catalog run makes
    thousands of REST calls, transient network hiccups shouldn't kill a study.
    `no_retry_statuses` skips the backoff loop for HTTP codes that are
    deterministic (e.g. 404), not transient."""
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_exc = e
            if isinstance(e, requests.HTTPError) and e.response is not None \
                    and e.response.status_code in no_retry_statuses:
                break
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_exc


class JsonDiskCache:
    """Lazy-loaded, disk-backed dict cache: get/set with a value factory,
    persisted to `path` on every write. Small wrapper around load_json/
    save_json so callers don't each re-implement the same lazy-load-then-
    save-on-write pattern."""

    def __init__(self, path):
        self.path = Path(path)
        self._data = None

    def _ensure_loaded(self):
        if self._data is None:
            self._data = load_json(self.path)
        return self._data

    def __contains__(self, key):
        return key in self._ensure_loaded()

    def get(self, key, default=None):
        return self._ensure_loaded().get(key, default)

    def set(self, key, value):
        data = self._ensure_loaded()
        data[key] = value
        save_json(data, self.path)
