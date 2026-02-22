"""
Properties.lk Property Data Scraper
=====================================
Scrapes property listings from properties.lk and saves to CSV.
Target: 5,000 records.

Categories scraped (in order until target is met):
  1. Apartment for Sale   (~1,206 ads)
  2. Apartment for Rent
  3. House for Sale        (largest category)
  4. House for Rent

The site is a JavaScript SPA, so Selenium is used for rendering.

Usage
-----
    python scraper.py                  # default: headless, 5 000 records
    python scraper.py --no-headless    # show the browser window
    python scraper.py --target 1000    # only scrape 1 000 records
"""

import argparse
import csv
import os
import sys
import time
import re
import logging
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
)
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# ─── Configuration ──────────────────────────────────────────────────
BASE_URL = "https://properties.lk"

# Scraped in this order; stop as soon as TARGET is reached.
CATEGORY_SLUGS = [
    "apartmentforsale",
    "apartmentforrent",
    "houseforsale",
    "houseforrent",
]

OUTPUT_CSV = "apartment_data.csv"
ADS_PER_PAGE = 26           # observed constant
DETAIL_DELAY = 1.0          # seconds between detail-page loads
PAGE_DELAY = 2.5            # seconds between listing pages
MAX_CONSECUTIVE_EMPTY = 5   # stop a category after this many blanks
MAX_RETRIES = 3             # per-page / per-detail retries

CSV_COLUMNS = [
    "ad_id",
    "title",
    "property_type",
    "listing_type",
    "location",
    "district",
    "detailed_address",
    "price_lkr",
    "price_type",
    "bedrooms",
    "bathrooms",
    "land_size",
    "land_size_unit",
    "property_size_sqft",
    "description",
    "posted_date",
    "posted_by",
    "url",
    "scraped_at",
]

# ─── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("scraper.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─── Scraper class ──────────────────────────────────────────────────
class PropertiesScraper:
    """Selenium-based scraper for properties.lk."""

    def __init__(self, target: int = 5_000, headless: bool = True):
        self.target = target
        self.headless = headless
        self.driver = None
        self.records: list[dict] = []
        self.scraped_ids: set[str] = set()
        self._resume_from_csv()

    # ── resume / csv ─────────────────────────────────────────────────
    def _resume_from_csv(self):
        if not os.path.exists(OUTPUT_CSV):
            return
        with open(OUTPUT_CSV, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.scraped_ids.add(row.get("ad_id", ""))
                self.records.append(row)
        log.info("Resumed %d records from %s", len(self.records), OUTPUT_CSV)

    def _write_header_if_needed(self):
        if not os.path.exists(OUTPUT_CSV):
            with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()

    def _append_csv(self, rec: dict):
        self._write_header_if_needed()
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writerow(rec)

    def _save_all(self):
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(self.records)
        log.info("Final save -> %s  (%d rows)", OUTPUT_CSV, len(self.records))

    # ── driver management ────────────────────────────────────────────
    def _start_driver(self):
        opts = Options()
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--disable-extensions")
        opts.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
        svc = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=svc, options=opts)
        self.driver.set_page_load_timeout(45)
        log.info("Chrome started (headless=%s)", self.headless)

    def _stop_driver(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
            log.info("Chrome stopped")

    def _restart_driver(self):
        """Kill and re-create the driver (recover from crashes)."""
        log.warning("Restarting Chrome driver ...")
        self._stop_driver()
        time.sleep(2)
        self._start_driver()

    # ── navigation helpers ───────────────────────────────────────────
    def _safe_get(self, url: str, wait: float = PAGE_DELAY) -> bool:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self.driver.get(url)
                time.sleep(wait)
                return True
            except WebDriverException as exc:
                log.warning("GET %s attempt %d failed: %s", url, attempt, exc)
                if attempt < MAX_RETRIES:
                    self._restart_driver()
                time.sleep(3)
        return False

    def _wait_for_ads(self, timeout: int = 12) -> bool:
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'a[href*="/allads/adverts/"]')
                )
            )
            time.sleep(0.5)
            return True
        except TimeoutException:
            return False

    # ── listing-page link collector ──────────────────────────────────
    def _collect_links(self) -> list[dict]:
        seen, out = set(), []
        for el in self.driver.find_elements(
            By.CSS_SELECTOR, 'a[href*="/allads/adverts/"]'
        ):
            href = el.get_attribute("href") or ""
            m = re.search(r"/adverts/(\d+)", href)
            if not m:
                continue
            aid = m.group(1)
            if aid in self.scraped_ids or aid in seen:
                continue
            seen.add(aid)
            out.append({"url": href, "ad_id": aid, "text": el.text.strip()})
        return out

    # ── detail-page parser ───────────────────────────────────────────
    def _parse_detail(self, url: str, ad_id: str) -> dict | None:
        if not self._safe_get(url, wait=DETAIL_DELAY):
            return None
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(0.5)
        except TimeoutException:
            return None

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        rec = {c: "" for c in CSV_COLUMNS}
        rec["ad_id"] = ad_id
        rec["url"] = url
        rec["scraped_at"] = datetime.now().isoformat()

        # title + property type + listing type + location
        for h2 in soup.find_all("h2"):
            t = h2.get_text(strip=True)
            if not re.search(r"for (Sale|Rent|Lease)", t, re.I):
                continue
            rec["title"] = t
            low = t.lower()
            for pt in ("Apartment", "House", "Land", "Commercial"):
                if pt.lower() in low:
                    rec["property_type"] = pt
                    break
            for lt in ("Sale", "Rent", "Lease"):
                if lt.lower() in low:
                    rec["listing_type"] = lt
                    break
            loc = re.search(r"in\s+(.+)$", t)
            if loc:
                rec["location"] = loc.group(1).strip()
            break

        # price
        pm = re.search(r"LKR\s*([\d,]+)", text)
        if pm:
            rec["price_lkr"] = pm.group(1).replace(",", "")
        for pt in ("total price", "per month", "per perch", "per sq ft"):
            if pt in text.lower():
                rec["price_type"] = pt
                break

        # district
        dm = re.search(
            r"(?:Apartments?|Houses?|Lands?|Commercial[\w ]*)\s+for\s+"
            r"(?:Sale|Rent|Lease)\s*[-\u2013]\s*(\w[\w ]*)",
            text,
        )
        if dm:
            rec["district"] = dm.group(1).strip()

        # bedrooms / bathrooms
        bm = re.search(r"Bedrooms?\s*[:\s]*(\d+)", text, re.I)
        if bm:
            rec["bedrooms"] = bm.group(1)
        bm2 = re.search(r"Bathrooms?\s*[:\s]*(\d+)", text, re.I)
        if bm2:
            rec["bathrooms"] = bm2.group(1)

        # land size
        lm = re.search(
            r"Land\s*Size\s*[-:]\s*([\d.]+)\s*(Perch(?:es)?|Acres?|sq\s*ft)?",
            text, re.I,
        )
        if lm:
            rec["land_size"] = lm.group(1)
            rec["land_size_unit"] = (lm.group(2) or "Perches").strip()

        # property size
        sm = re.search(
            r"(?:House|Property|Apartment)\s*Size\s*[-:]\s*([\d,]+)\s*sq\s*ft",
            text, re.I,
        )
        if sm:
            rec["property_size_sqft"] = sm.group(1).replace(",", "")

        # posted date / by
        pdm = re.search(r"Posted\s+on\s*:\s*(.+?)(?:,\s*[A-Z])", text)
        if pdm:
            rec["posted_date"] = pdm.group(1).strip()
        pbm = re.search(r"Posted\s+by:\s*(.+?)(?:\n|$)", text)
        if pbm:
            rec["posted_by"] = pbm.group(1).strip()

        # detailed address
        am = re.search(r"Posted\s+on\s*:.+?,\s*(.+?)$", text, re.M)
        if am:
            rec["detailed_address"] = am.group(1).strip()

        # description
        dm2 = re.search(
            r"Description\s*\n(.*?)(?=Posted\s+by:|Share\s+this\s+Ad|Similar\s+Properties)",
            text, re.S,
        )
        if dm2:
            rec["description"] = dm2.group(1).strip()[:2000]

        return rec

    # ── per-category scraper ─────────────────────────────────────────
    def _scrape_category(self, slug: str):
        """Paginate through one category and scrape every detail page."""
        cat_url = f"{BASE_URL}/allads?category={slug}"
        page = 1
        empties = 0

        while len(self.records) < self.target:
            url = f"{cat_url}&page={page}"
            log.info("[%s] listing page %d ...", slug, page)

            if not self._safe_get(url):
                page += 1
                empties += 1
                if empties >= MAX_CONSECUTIVE_EMPTY:
                    break
                continue

            if not self._wait_for_ads():
                empties += 1
                if empties >= MAX_CONSECUTIVE_EMPTY:
                    log.info("[%s] %d empty pages - done with category", slug, empties)
                    break
                page += 1
                continue

            empties = 0
            links = self._collect_links()
            if not links:
                page += 1
                continue

            log.info("[%s] page %d -> %d new links", slug, page, len(links))

            for lnk in links:
                if len(self.records) >= self.target:
                    return
                rec = self._parse_detail(lnk["url"], lnk["ad_id"])
                if rec and rec.get("title"):
                    self.records.append(rec)
                    self.scraped_ids.add(lnk["ad_id"])
                    self._append_csv(rec)
                    if len(self.records) % 25 == 0:
                        log.info(
                            ">>> %d / %d  (%.1f%%)",
                            len(self.records),
                            self.target,
                            len(self.records) / self.target * 100,
                        )
            page += 1

    # ── public entry ─────────────────────────────────────────────────
    def run(self):
        log.info(
            "=== Properties.lk Scraper  target=%d  existing=%d ===",
            self.target, len(self.records),
        )
        if len(self.records) >= self.target:
            log.info("Already have %d records - nothing to do.", len(self.records))
            return

        try:
            self._start_driver()

            for slug in CATEGORY_SLUGS:
                if len(self.records) >= self.target:
                    break
                log.info("-- Category: %s --", slug)
                self._scrape_category(slug)
                log.info("After [%s]: %d records total", slug, len(self.records))

        except KeyboardInterrupt:
            log.info("Interrupted - saving progress ...")
        except Exception:
            log.exception("Fatal error")
        finally:
            self._stop_driver()
            if self.records:
                self._save_all()
            log.info(
                "=== Done. %d records saved to %s ===",
                len(self.records), OUTPUT_CSV,
            )


# ─── CLI ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Scrape properties.lk")
    ap.add_argument(
        "--target", type=int, default=5000,
        help="Number of records to collect (default 5000)",
    )
    ap.add_argument(
        "--no-headless", action="store_true",
        help="Show the browser window",
    )
    args = ap.parse_args()

    scraper = PropertiesScraper(
        target=args.target,
        headless=not args.no_headless,
    )
    scraper.run()


if __name__ == "__main__":
    main()
