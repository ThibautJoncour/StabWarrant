"""SG Bourse NVIDIA call/put listing support for the Stability Shiny application.

SG site HTML/URLs can change. This module never supplies fabricated quotes,
never labels a partial listing as exhaustive, and never turns an error into XLSX.
"""
from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup

SG_WARRANTS_URL = (
    "https://sgbourse.fr/product-search/warrants/"
    "actions%2520am%C3%A9ricaines/nvidia/all-assets?ShowFilter=false"
)
COLS = ["Mnémo", "ISIN", "Type", "Sous-jacent", "Strike_USD", "Maturité", "Parité", "Bid_EUR", "Ask_EUR", "URL_produit", "Source", "Date_extraction_UTC"]


def _number(value):
    if value is None or pd.isna(value):
        return None
    s = str(value).replace("\u202f", "").replace("\xa0", "").replace(" ", "")
    m = re.search(r"-?\d[\d.,]*", s)
    if not m:
        return None
    s = m.group()
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _date(value):
    d = pd.to_datetime(value, dayfirst=True, errors="coerce")
    return None if pd.isna(d) else d.strftime("%Y-%m-%d")


def _type(text):
    if re.search(r"\bput\b", str(text), re.I):
        return "Put"
    if re.search(r"\bcall\b", str(text), re.I):
        return "Call"
    return None


def _clean_frame(rows, source):
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=COLS)
    out = out.copy()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["Source"] = source
    out["Date_extraction_UTC"] = now
    for col in COLS:
        if col not in out:
            out[col] = None
    out = out[COLS]
    out = out[out["Type"].isin(["Call", "Put"]) & out["Strike_USD"].notna() & out["Maturité"].notna()]
    out = out.drop_duplicates(subset=["Mnémo", "ISIN"], keep="first").reset_index(drop=True)
    return out


def parse_sg_listing_html(html):
    """Parse complete product rows from a SG NVIDIA warrants listing page.

    If SG serves a client-side JS shell, produce an explicit empty result rather
    than quietly inventing instruments, market prices or Greeks.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.select("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
        if len(cells) < 4:
            continue
        entire = " ".join(cells)
        # Required to avoid exporting unrelated assets from a general listing.
        if not re.search(r"\bNVIDIA\b", entire, re.I):
            continue
        kind = _type(entire)
        expiry_match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", entire)
        strike_match = re.search(r"(\d[\d.,\u00a0\u202f ]*)\s*USD", entire, re.I)
        parity_match = re.search(r"\b(\d+)\s*:\s*(\d+)\b", entire)
        if not (kind and expiry_match and strike_match and parity_match):
            continue
        product_anchor = tr.select_one('a[href*="product-details"]')
        link = (product_anchor.get("href", "") if product_anchor else "").strip()
        if link.startswith("/"):
            link = "https://sgbourse.fr" + link
        if link and not link.startswith("https://sgbourse.fr") and not link.startswith("https://www.sgbourse.fr"):
            link = ""
        memo = (product_anchor.get_text(" ", strip=True) if product_anchor else "")
        if not re.fullmatch(r"[A-Z0-9]{4,8}", memo.upper()):
            memo = next((x.strip() for x in cells if re.fullmatch(r"[A-Za-z0-9]{4,8}", x.strip())), "")
        if not memo:
            continue
        isin_match = re.search(r"\bDE[A-Z0-9]{10}\b", entire)
        strike = _number(strike_match.group(1))
        parity_a, parity_b = int(parity_match.group(1)), int(parity_match.group(2))
        if parity_a <= 0 or parity_b <= 0 or strike is None:
            continue
        rows.append({
            "Mnémo": memo.upper(), "ISIN": isin_match.group() if isin_match else None,
            "Type": kind, "Sous-jacent": "NVIDIA", "Strike_USD": strike,
            "Maturité": _date(expiry_match.group()), "Parité": f"{parity_a}:{parity_b}",
            # Do not infer quotes from unlabelled table cells.
            "Bid_EUR": None, "Ask_EUR": None, "URL_produit": link or None,
        })
    return _clean_frame(rows, SG_WARRANTS_URL)


def download_listing(timeout=25):
    req = Request(SG_WARRANTS_URL, headers={
        "User-Agent": "Mozilla/5.0 (compatible; StabilityResearch/1.0)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "fr-FR,fr;q=0.9",
    })
    with urlopen(req, timeout=timeout) as res:
        html = res.read(12_000_000)
        charset = res.headers.get_content_charset() or "utf-8"
    data = parse_sg_listing_html(html.decode(charset, errors="replace"))
    if data.empty:
        raise ValueError("SG n'a renvoyé aucune ligne NVIDIA Call/Put exploitable (page JS ou structure modifiée). Utilisez l'import Excel/CSV.")
    return data


def import_sg_warrant_file(path, filename):
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    elif filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype=str)
    else:
        raise ValueError("Format attendu : .csv, .xls ou .xlsx")
    synonyms = {
        "mémo": "Mnémo", "mnémo": "Mnémo", "mnémo.": "Mnémo", "code": "Mnémo", "mnemo": "Mnémo",
        "isin": "ISIN", "code isin": "ISIN", "type": "Type", "type de produit": "Type",
        "sous-jacent": "Sous-jacent", "sous jacent": "Sous-jacent", "nom de l'actif": "Sous-jacent", "actif": "Sous-jacent",
        "strike": "Strike_USD", "prix d'exercice": "Strike_USD", "strike_usd": "Strike_USD",
        "maturité": "Maturité", "maturite": "Maturité", "parité": "Parité", "parite": "Parité",
        "achat": "Bid_EUR", "vente": "Ask_EUR", "bid": "Bid_EUR", "ask": "Ask_EUR",
        "bid_eur": "Bid_EUR", "ask_eur": "Ask_EUR", "url_produit": "URL_produit",
    }
    df = df.rename(columns={c: synonyms.get(str(c).strip().lower(), str(c).strip()) for c in df.columns})
    required = {"Mnémo", "Type", "Strike_USD", "Maturité", "Parité"}
    if not required.issubset(df.columns):
        raise ValueError("Colonnes nécessaires : Mnémo, Type, Prix d'exercice/Strike, Maturité, Parité")
    results = []
    for _, row in df.iterrows():
        kind = _type(row["Type"])
        underlier = str(row.get("Sous-jacent", "NVIDIA"))
        if "Sous-jacent" in df and "NVIDIA" not in underlier.upper():
            continue
        strike = _number(row["Strike_USD"])
        maturity = _date(row["Maturité"])
        parity = str(row["Parité"]).strip() if pd.notna(row["Parité"]) else ""
        if not kind or strike is None or not maturity or not parity:
            continue
        results.append({
            "Mnémo": str(row["Mnémo"]).strip(), "ISIN": row.get("ISIN"),
            "Type": kind, "Sous-jacent": "NVIDIA", "Strike_USD": strike,
            "Maturité": maturity, "Parité": parity,
            "Bid_EUR": _number(row.get("Bid_EUR")), "Ask_EUR": _number(row.get("Ask_EUR")),
            "URL_produit": row.get("URL_produit"),
        })
    out = _clean_frame(results, f"Fichier utilisateur : {filename}")
    if out.empty:
        raise ValueError("Aucun warrant Call/Put NVIDIA valide dans ce fichier.")
    return out


def csv_bytes(frame):
    if frame.empty:
        raise ValueError("Aucune donnée de warrant à exporter.")
    return frame.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
