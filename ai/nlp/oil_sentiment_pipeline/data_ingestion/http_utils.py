"""HTTP avec retry et backoff exponentiel."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


def get_with_retry(
    url: str,
    *,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    timeout: int = 15,
    max_retries: int = 3,
    backoff_sec: float = 1.0,
) -> requests.Response:
    """GET avec retries sur erreurs réseau et 429/5xx."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = backoff_sec * (2**attempt)
                logger.warning(
                    "HTTP %s pour %s — retry %d/%d dans %.1fs",
                    resp.status_code,
                    url[:80],
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            wait = backoff_sec * (2**attempt)
            logger.warning(
                "Erreur HTTP %s — retry %d/%d dans %.1fs",
                exc,
                attempt + 1,
                max_retries,
                wait,
            )
            time.sleep(wait)
    raise last_exc or RuntimeError(f"Échec GET après {max_retries} tentatives : {url}")
