"""Scraper for IMGT 2D structure detail pages.

Fetches and parses chain-level sequence/mutation data for antibodies
from https://imgt.org/3Dstructure-DB/cgi/details.cgi?pdbcode={inn_id}.
"""

import asyncio
import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import aiohttp
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import INNStructureCollection, INNStructureResult
from .parsers import parse_imgt_inn_summary_table

logger = logging.getLogger(__name__)


class IMGT2DStructureScraper:
    BASE_URL = "https://imgt.org/3Dstructure-DB/cgi/details.cgi"

    def __init__(self, jsession_id: str, php_session_id: str, strict_positions: bool = True):
        self.strict_positions = strict_positions
        self._cookies = {"JSESSIONID": jsession_id, "PHPSESSID": php_session_id}
        self._session = requests.Session()
        self._session.cookies.update(self._cookies)
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self._session.mount("https://", HTTPAdapter(max_retries=retry))

    # --- Sync single fetch ---

    def fetch(self, inn_id: str) -> Optional[INNStructureResult]:
        """Fetch and parse the 2D structure page for a single INN ID.
        Returns None if the INN has no entry in the 2D structure database."""
        resp = self._session.get(self.BASE_URL, params={"pdbcode": inn_id})
        resp.raise_for_status()
        return self._parse_detail_page(resp.text, str(inn_id))

    def _parse_detail_page(self, html: str, inn_id: str) -> Optional[INNStructureResult]:
        """Parse detail page. Returns None if the INN has no 2D structure entry."""
        soup = BeautifulSoup(html, "html.parser")
        main_div = soup.find("div", id="main")
        if not main_div:
            return None

        tables = main_div.find_all("table")
        all_chains = []

        _titre_re = re.compile(r"^titre_(.+)$")

        for idx, table in enumerate(tables):
            chain_type = None
            for td in table.find_all("td"):
                for cls in td.get("class") or []:
                    m = _titre_re.match(cls)
                    if m:
                        chain_type = m.group(1)
                        break
                if chain_type:
                    break
            if chain_type is None:
                raise ValueError(f"Table {idx} for INN {inn_id}: no chain type detected")

            chains = parse_imgt_inn_summary_table(table, chain_type, inn_id, self.strict_positions)
            all_chains.extend(chains)

        return INNStructureResult(inn_id=inn_id, chains=all_chains)

    # --- Async bulk fetch ---

    async def _fetch_async(
        self,
        session: aiohttp.ClientSession,
        inn_id: str,
        semaphore: asyncio.Semaphore,
        results: dict,
        missing: list,
        failures: dict,
        pbar,
    ) -> None:
        async with semaphore:
            try:
                url = f"{self.BASE_URL}?pdbcode={inn_id}"
                async with session.get(url) as resp:
                    html = await resp.text()
                    parsed = self._parse_detail_page(html, str(inn_id))
                    if parsed is None:
                        missing.append(inn_id)
                    else:
                        results[inn_id] = parsed
            except Exception as e:
                logger.warning(f"Failed {inn_id}: {e}")
                failures[inn_id] = str(e)
            finally:
                pbar.update(1)

    async def _fetch_all_async(
        self,
        inn_ids: list[str],
        concurrency: int,
        results: dict,
        missing: list,
        failures: dict,
    ) -> None:
        from tqdm.asyncio import tqdm

        semaphore = asyncio.Semaphore(concurrency)
        pbar = tqdm(total=len(inn_ids), desc="Fetching IMGT 2D structures")
        async with aiohttp.ClientSession(cookies=self._cookies) as session:
            tasks = [self._fetch_async(session, iid, semaphore, results, missing, failures, pbar) for iid in inn_ids]
            await asyncio.gather(*tasks)
        pbar.close()

    def fetch_all(
        self,
        inn_ids: list[str],
        concurrency: int = 10,
        limit: Optional[int] = None,
        checkpoint_path: Optional[Path] = None,
    ) -> INNStructureCollection:
        """Bulk fetch 2D structure pages with async concurrency.

        Args:
            inn_ids: List of IMGT INN IDs to fetch.
            concurrency: Max parallel requests.
            limit: Cap the number of IDs to fetch.
            checkpoint_path: If provided, save/resume progress from this pickle file.
        """
        targets = [str(i) for i in inn_ids]
        if limit is not None:
            targets = targets[:limit]

        # Resume from checkpoint
        results: dict[str, INNStructureResult] = {}
        missing: list[str] = []
        failures: dict[str, str] = {}
        if checkpoint_path and checkpoint_path.exists():
            with open(checkpoint_path, "rb") as f:
                saved = pickle.load(f)
            results = saved.get("results", {})
            missing = saved.get("missing", [])
            failures = saved.get("failures", {})
            logger.info(f"Resumed {len(results)} results from checkpoint")

        skip = set(results) | set(missing) | set(failures)
        remaining = [i for i in targets if i not in skip]

        if remaining:
            try:
                loop = asyncio.get_running_loop()
                import nest_asyncio

                nest_asyncio.apply(loop)
            except RuntimeError:
                pass
            asyncio.run(self._fetch_all_async(remaining, concurrency, results, missing, failures))

        # Save checkpoint
        if checkpoint_path:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({"results": results, "missing": missing, "failures": failures}, f)

        return INNStructureCollection(
            results=list(results.values()),
            missing=missing,
            failures=failures,
        )

    def __repr__(self) -> str:
        return "IMGT2DStructureScraper()"
