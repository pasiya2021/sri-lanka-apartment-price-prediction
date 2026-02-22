"""
scrape.py
=========
Lightweight, requests + BeautifulSoup scraper for
https://properties.lk/allads?category=apartmentforsale

ETHICAL SCRAPING NOTE
---------------------
•  This script is intended for **educational and academic purposes only**.
•  It respects robots.txt – the site's robots.txt is checked before
   scraping begins.  If disallowed, the script will abort.
•  A configurable delay (default 2 s) is inserted between every HTTP
   request to avoid putting excessive load on the server.
•  The User‑Agent header clearly identifies the scraper.
•  Do NOT use this script for commercial data harvesting without explicit
   permission from properties.lk.

NOTE: properties.lk is a JavaScript Single‑Page Application (SPA).
Plain HTTP requests only receive a ~473‑byte shell without data.
For real scraping you MUST use the Selenium‑based ``scraper.py``
that ships with this project.  This file fulfils the assignment
requirement of a requests + BS4 scraper and will work if the site
ever adds server‑side rendering (SSR).

Usage
-----
    python scrape.py
    python scrape.py --max-pages 10 --output data/my_data.csv --delay 3
"""

import argparse
import csv
import os
import re
import sys
import time
import logging
from datetime import datetime
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ─── Configuration ──────────────────────────────────────────────────
BASE_URL = "https://properties.lk"
LISTING_URL = f"{BASE_URL}/allads"
DEFAULT_MAX_PAGES = 50
DEFAULT_OUTPUT = "apartment_data.csv"
DEFAULT_DELAY = 2.0        # seconds between requests
USER_AGENT = (
    "SriLankaApartmentScraper/1.0 "
    "(Academic project; contact: student@university.lk)"
)

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scrape")

# ─── Robots.txt check ───────────────────────────────────────────────

def check_robots(url: str = BASE_URL) -> bool:
    """Return True if we're allowed to scrape the listing pages."""
    rp = RobotFileParser()
    rp.set_url(f"{url}/robots.txt")
    try:
        rp.read()
    except Exception:
        log.warning("Could not fetch robots.txt – proceeding cautiously.")
        return True
    allowed = rp.can_fetch(USER_AGENT, f"{url}/allads")
    if not allowed:
        log.error("robots.txt DISALLOWS scraping /allads – aborting.")
    return allowed


# ─── HTTP session ────────────────────────────────────────────────────

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


# ─── Parsers ─────────────────────────────────────────────────────────

def parse_listing_page(html: str) -> list[dict]:
    """
    Extract ad links and preview text from a listing page.
    Returns a list of dicts with keys: ad_id, url, text.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a", href=re.compile(r"/allads/adverts/\d+"))
    seen, results = set(), []
    for a in links:
        m = re.search(r"/adverts/(\d+)", a["href"])
        if not m or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        results.append({
            "ad_id": m.group(1),
            "url":   BASE_URL + a["href"] if a["href"].startswith("/") else a["href"],
            "text":  a.get_text(strip=True),
        })
    return results


def parse_detail_page(html: str, ad_id: str, url: str) -> dict:
    """Parse a single ad's detail page into a flat dict."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)

    rec = {c: "" for c in CSV_COLUMNS}
    rec["ad_id"] = ad_id
    rec["url"] = url
    rec["scraped_at"] = datetime.now().isoformat()

    # title / type / listing
    for h2 in soup.find_all("h2"):
        t = h2.get_text(strip=True)
        if not re.search(r"for (Sale|Rent|Lease)", t, re.I):
            continue
        rec["title"] = t
        for pt in ("Apartment", "House", "Land", "Commercial"):
            if pt.lower() in t.lower():
                rec["property_type"] = pt; break
        for lt in ("Sale", "Rent", "Lease"):
            if lt.lower() in t.lower():
                rec["listing_type"] = lt; break
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
            rec["price_type"] = pt; break

    # district
    dm = re.search(
        r"(?:Apartments?|Houses?)\s+for\s+(?:Sale|Rent|Lease)\s*[-\u2013]\s*(\w[\w ]*)",
        text)
    if dm:
        rec["district"] = dm.group(1).strip()

    # beds / baths
    bm = re.search(r"Bedrooms?\s*[:\s]*(\d+)", text, re.I)
    if bm: rec["bedrooms"] = bm.group(1)
    bm2 = re.search(r"Bathrooms?\s*[:\s]*(\d+)", text, re.I)
    if bm2: rec["bathrooms"] = bm2.group(1)

    # sizes
    lm = re.search(r"Land\s*Size\s*[-:]\s*([\d.]+)\s*(Perch(?:es)?|Acres?)?", text, re.I)
    if lm:
        rec["land_size"] = lm.group(1)
        rec["land_size_unit"] = (lm.group(2) or "Perches").strip()
    sm = re.search(r"(?:House|Apartment|Property)\s*Size\s*[-:]\s*([\d,]+)\s*sq\s*ft", text, re.I)
    if sm:
        rec["property_size_sqft"] = sm.group(1).replace(",", "")

    # posted info
    pdm = re.search(r"Posted\s+on\s*:\s*(.+?)(?:,\s*[A-Z])", text)
    if pdm: rec["posted_date"] = pdm.group(1).strip()
    pbm = re.search(r"Posted\s+by:\s*(.+?)(?:\n|$)", text)
    if pbm: rec["posted_by"] = pbm.group(1).strip()
    am = re.search(r"Posted\s+on\s*:.+?,\s*(.+?)$", text, re.M)
    if am: rec["detailed_address"] = am.group(1).strip()

    # description
    dm2 = re.search(
        r"Description\s*\n(.*?)(?=Posted\s+by:|Share\s+this|Similar\s+Properties)",
        text, re.S)
    if dm2:
        rec["description"] = dm2.group(1).strip()[:2000]

    return rec


# ─── Main scrape loop ───────────────────────────────────────────────

def scrape(max_pages: int, output_csv: str, delay: float):
    if not check_robots():
        sys.exit(1)

    session = make_session()

    # Write header
    write_header = not os.path.exists(output_csv)
    if write_header:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

    total_scraped = 0

    for page in tqdm(range(1, max_pages + 1), desc="Listing pages"):
        params = {"category": "apartmentforsale", "page": page}
        try:
            r = session.get(LISTING_URL, params=params, timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            log.warning("Page %d failed: %s", page, e)
            time.sleep(delay)
            continue

        ads = parse_listing_page(r.text)
        if not ads:
            log.info("Page %d returned 0 ads – stopping.", page)
            break

        for ad in ads:
            time.sleep(delay)
            try:
                dr = session.get(ad["url"], timeout=20)
                dr.raise_for_status()
            except requests.RequestException as e:
                log.warning("Detail %s failed: %s", ad["ad_id"], e)
                continue

            rec = parse_detail_page(dr.text, ad["ad_id"], ad["url"])
            if rec.get("title"):
                with open(output_csv, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(rec)
                total_scraped += 1

        log.info("Page %d done – %d ads collected  (total: %d)",
                 page, len(ads), total_scraped)
        time.sleep(delay)

    log.info("Scraping finished. %d records saved → %s", total_scraped, output_csv)


# ─── CLI ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Scrape apartment listings from properties.lk")
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                    help="Maximum number of listing pages to scrape")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="Output CSV file path")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                    help="Delay (seconds) between HTTP requests")
    args = ap.parse_args()

    scrape(args.max_pages, args.output, args.delay)


if __name__ == "__main__":
    main()
