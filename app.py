from shiny import App, ui, render, reactive
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import Request, urlopen


# ==========================================================
# WARRANTS NVIDIA — IMPORT / EXPORT SG
# ==========================================================
SG_WARRANTS_URL = (
    "https://sgbourse.fr/product-search/warrants/"
    "actions%2520am%C3%A9ricaines/nvidia/all-assets?ShowFilter=false"
)
SG_CALL_PUT_EXPORT_URL = (
    "https://www.sgbourse.fr/EmcWebApi/api/ProductSearch/Export"
    "?PageNum=1&ProductClassificationId=42"
    "&AssetTypeId=1&AssetTypeMenuId=33&AssetId=320"
)
WARRANT_COLS = [
    "Mnémo", "ISIN", "Type", "Sous-jacent", "Strike_USD", "Maturité",
    "Parité", "Bid_EUR", "Ask_EUR", "URL_produit",
    "Source", "Date_extraction_UTC",
]


def _warrant_number(value):
    if value is None or pd.isna(value):
        return None
    txt = str(value).replace("\u202f", "").replace("\xa0", "").replace(" ", "")
    m = re.search(r"-?\d[\d.,]*", txt)
    if not m:
        return None
    txt = m.group()
    if "," in txt:
        txt = txt.replace(".", "").replace(",", ".")
    try:
        return float(txt)
    except ValueError:
        return None


def _warrant_type(value):
    txt = str(value).lower()
    if re.search(r"\bput\b", txt):
        return "Put"
    if re.search(r"\bcall\b", txt):
        return "Call"
    return None


def _clean_warrants(rows, source):
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=WARRANT_COLS)
    out["Source"] = source
    out["Date_extraction_UTC"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for col in WARRANT_COLS:
        if col not in out:
            out[col] = None
    out = out[WARRANT_COLS]
    out = out[out["Type"].isin(["Call", "Put"])]
    out = out[out["Strike_USD"].notna() & out["Maturité"].notna()]
    return out.drop_duplicates(subset=["Mnémo", "ISIN"]).reset_index(drop=True)


def import_call_put(path, filename):
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    elif filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype=str)
    else:
        raise ValueError("Formats autorisés : CSV, XLS ou XLSX.")

    synonyms = {
        "mémo": "Mnémo", "mnemo": "Mnémo", "mnémo": "Mnémo",
        "mnémo.": "Mnémo", "code": "Mnémo",
        "isin": "ISIN", "code isin": "ISIN",
        "type": "Type", "type de produit": "Type",
        "sous-jacent": "Sous-jacent", "sous jacent": "Sous-jacent",
        "nom de l'actif": "Sous-jacent", "actif": "Sous-jacent",
        "strike": "Strike_USD", "prix d'exercice": "Strike_USD",
        "strike_usd": "Strike_USD",
        "maturité": "Maturité", "maturite": "Maturité",
        "parité": "Parité", "parite": "Parité",
        "achat": "Bid_EUR", "bid": "Bid_EUR", "bid_eur": "Bid_EUR",
        "vente": "Ask_EUR", "ask": "Ask_EUR", "ask_eur": "Ask_EUR",
        "url_produit": "URL_produit",
    }
    df = df.rename(columns={
        c: synonyms.get(str(c).strip().lower(), str(c).strip())
        for c in df.columns
    })
    required = {"Mnémo", "Type", "Strike_USD", "Maturité", "Parité"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Colonnes requises absentes : {missing}")

    rows = []
    for _, row in df.iterrows():
        if "Sous-jacent" in df and "NVIDIA" not in str(row.get("Sous-jacent", "")).upper():
            continue
        kind = _warrant_type(row["Type"])
        strike = _warrant_number(row["Strike_USD"])
        maturity = pd.to_datetime(row["Maturité"], dayfirst=True, errors="coerce")
        parity = str(row["Parité"]).strip() if pd.notna(row["Parité"]) else ""
        if not kind or strike is None or pd.isna(maturity) or not parity:
            continue
        rows.append({
            "Mnémo": str(row["Mnémo"]).strip(),
            "ISIN": row.get("ISIN"),
            "Type": kind,
            "Sous-jacent": "NVIDIA",
            "Strike_USD": strike,
            "Maturité": maturity.strftime("%Y-%m-%d"),
            "Parité": parity,
            "Bid_EUR": _warrant_number(row.get("Bid_EUR")),
            "Ask_EUR": _warrant_number(row.get("Ask_EUR")),
            "URL_produit": row.get("URL_produit"),
        })
    out = _clean_warrants(rows, f"Import utilisateur : {filename}")
    if out.empty:
        raise ValueError("Aucun Call/Put NVIDIA exploitable dans le fichier.")
    return out


class _SGTableParser(HTMLParser):
    """Extraction prudente des lignes de tableau HTML serveur."""
    def __init__(self):
        super().__init__()
        self.rows = []
        self._row = None
        self._cell = None
        self._link = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell = []
        elif tag == "a" and self._row is not None:
            href = attrs.get("href", "")
            if "product-details" in href:
                self._link = href

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self._cell is not None:
            self._row.append(" ".join(" ".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append((self._row, self._link))
            self._row = None
            self._link = None


def download_call_put():
    req = Request(SG_WARRANTS_URL, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "fr-FR,fr;q=0.9",
    })
    with urlopen(req, timeout=25) as res:
        html = res.read(12_000_000).decode(
            res.headers.get_content_charset() or "utf-8", errors="replace"
        )
    parser = _SGTableParser()
    parser.feed(html)
    rows = []
    for cells, link in parser.rows:
        txt = " ".join(cells)
        if not re.search(r"\bNVIDIA\b", txt, re.I):
            continue
        kind = _warrant_type(txt)
        expiry = re.search(r"\b\d{2}/\d{2}/\d{4}\b", txt)
        strike_match = re.search(r"(\d[\d.,\u00a0\u202f ]*)\s*USD", txt, re.I)
        parity_match = re.search(r"\b(\d+)\s*:\s*(\d+)\b", txt)
        memo = next((x.strip() for x in cells
                     if re.fullmatch(r"[A-Za-z0-9]{4,8}", x.strip())), None)
        if not all([kind, expiry, strike_match, parity_match, memo]):
            continue
        if int(parity_match.group(1)) <= 0 or int(parity_match.group(2)) <= 0:
            continue
        rows.append({
            "Mnémo": memo.upper(),
            "ISIN": (re.search(r"\bDE[A-Z0-9]{10}\b", txt) or [None])[0],
            "Type": kind,
            "Sous-jacent": "NVIDIA",
            "Strike_USD": _warrant_number(strike_match.group(1)),
            "Maturité": pd.to_datetime(expiry.group(), dayfirst=True).strftime("%Y-%m-%d"),
            "Parité": f"{parity_match.group(1)}:{parity_match.group(2)}",
            "Bid_EUR": None,
            "Ask_EUR": None,
            "URL_produit": ("https://sgbourse.fr" + link if link and link.startswith("/") else link),
        })
    out = _clean_warrants(rows, SG_WARRANTS_URL)
    if out.empty:
        raise ValueError(
            "Le site SG n'a fourni aucune ligne Call/Put exploitable : "
            "page JavaScript ou structure modifiée. Utilisez l'import CSV/Excel."
        )
    return out

# ==========================================================
# UI
# ==========================================================
app_ui = ui.page_fluid(
    ui.h2("Stability / Double No Touch Pricer"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_numeric("spot", "Spot", 225.51),
            ui.input_numeric("lower", "Barrière basse", 200),
            ui.input_numeric("upper", "Barrière haute", 250),
            ui.input_numeric("sigma", "Volatilité", 0.33),
            ui.input_numeric("rate", "Taux", 0.04),
            ui.input_numeric("days", "Jours jusqu'à maturité", 85),
            ui.input_numeric("notional", "Nominal", 10),
            ui.input_numeric("npaths", "Nb trajectoires", 100000),
            ui.download_button(
                "download_sg",
                "Télécharger export SG",
            ),
            ui.input_file(
                "sg_file",
                "Importer export SG",
                accept=[".xlsx", ".xls"],
            ),
            ui.download_button("download_call_put_sg", "Télécharger Call/Put SG NVIDIA (Excel)"),
            ui.input_file(
                "warrants_file", "Importer Call/Put NVIDIA (Excel ou CSV)",
                accept=[".xlsx", ".xls", ".csv"],
            ),
            ui.download_button("download_warrants", "Exporter Call/Put (CSV)"),
            ui.input_action_button("run", "Calculer"),
            width=320,
        ),
        ui.navset_tab(
            ui.nav_panel(
                "Pricer",
                ui.output_text_verbatim("results"),
                ui.output_plot("vol_chart"),
            ),
            ui.nav_panel(
                "Batch SG",
                ui.output_data_frame("greeks_batch"),
            ),
            ui.nav_panel(
                "Pur Short Vol",
                ui.p(
                    "Priorité : barrière la plus éloignée, |Delta| faible et Vega négatif par euro de prime."
                ),
                ui.output_data_frame("short_vol_batch"),
            ),
            ui.nav_panel(
                "Warrants Call/Put NVDA",
                ui.p("Source : ", ui.a("SG Bourse NVIDIA", href=SG_WARRANTS_URL, target="_blank")),
                ui.p("1. Télécharger le fichier Excel SG. 2. L’importer manuellement ci-dessus pour afficher les warrants."),
                ui.output_text("warrants_status"),
                ui.output_data_frame("warrants_table"),
            ),
            ui.nav_panel(
                "Theta Carry",
                ui.p(
                    "Classement par Theta / Prime : carry journalier estimé rapporté au prix d'achat (Ask)."
                ),
                ui.output_data_frame("theta_carry_batch"),
            ),
        ),
    ),
)


# ==========================================================
# SERVER
# ==========================================================
def server(input, output, session):


    # ======================================================
    # TÉLÉCHARGEMENT DIRECT DE L'EXPORT SG BOURSE
    # ======================================================
    @output
    @render.download_button(filename="sg_stability_export.xlsx")
    def download_sg():
        url = (
            "https://www.sgbourse.fr/EmcWebApi/api/ProductSearch/Export"
            "?PageNum=1&ProductClassificationId=8&AssetId=320"
        )

        req = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36"
                ),
                "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/octet-stream,*/*",
            },
        )

        try:
            with urlopen(req, timeout=30) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk
        except Exception as exc:
            # Le téléchargement Shiny attend des bytes. On renvoie donc
            # un petit fichier texte explicatif en cas d'erreur réseau.
            yield (
                "Impossible de télécharger l'export SG Bourse.\n"
                f"Erreur : {exc}\n"
                f"URL : {url}\n"
            ).encode("utf-8")

    # ======================================================
    # CALL / PUT NVIDIA : récupération, import et export
    # ======================================================
    @reactive.calc
    def warrants_data():
        uploaded = input.warrants_file()
        if uploaded:
            item = uploaded[0]
            try:
                return import_call_put(item["datapath"], item["name"]), "Import utilisateur", None
            except Exception as exc:
                return pd.DataFrame(), "Import impossible", str(exc)
        return pd.DataFrame(), "Non chargé", None

    @output
    @render.download_button(filename="sg_nvidia_call_put.xlsx")
    def download_call_put_sg():
        # Export Excel officiel SG observé dans l'onglet Network du navigateur.
        req = Request(
            SG_CALL_PUT_EXPORT_URL,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": (
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet,application/vnd.ms-excel,"
                    "application/octet-stream,*/*"
                ),
                "Referer": SG_WARRANTS_URL,
            },
        )
        # Une erreur HTTP est remontée explicitement plutôt que de générer
        # un faux fichier Excel contenant une page d'erreur.
        with urlopen(req, timeout=40) as response:
            first = response.read(8192)
            if not (first.startswith(b"PK\x03\x04") or first.startswith(b"\xd0\xcf\x11\xe0")):
                raise ValueError(
                    "SG n'a pas renvoyé un fichier Excel valide. "
                    "Vérifiez le lien Export dans Network."
                )
            yield first
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk

    @output
    @render.text
    def warrants_status():
        data, source, error = warrants_data()
        if error:
            return f"{source} : {error}"
        if data.empty:
            return "Aucun Call/Put chargé. Téléchargez le fichier SG, puis importez-le en Excel ou CSV."
        return (f"{len(data)} warrants NVIDIA : "
                f"{(data['Type'] == 'Call').sum()} Calls et "
                f"{(data['Type'] == 'Put').sum()} Puts. {source}.")

    @output
    @render.data_frame
    def warrants_table():
        data, _, _ = warrants_data()
        return render.DataGrid(data, filters=True)

    @output
    @render.download_button(
        filename=lambda: f"sg_nvidia_call_put_{datetime.now():%Y%m%d_%H%M}.csv"
    )
    def download_warrants():
        data, _, error = warrants_data()
        if error or data.empty:
            raise ValueError(error or "Aucun Call/Put à exporter.")
        yield data.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")

    # ======================================================
    # Fonction Monte Carlo commune : Double No Touch
    # ======================================================
    def dnt_mc(
        S0,
        LOWER,
        UPPER,
        sigma,
        r,
        T,
        NOTIONAL,
        N_PATHS,
        random_seed=None,
    ):
        S0 = float(S0)
        LOWER = float(LOWER)
        UPPER = float(UPPER)
        sigma = float(sigma)
        r = float(r)
        T = float(T)
        NOTIONAL = float(NOTIONAL)
        N_PATHS = int(N_PATHS)

        if LOWER >= UPPER:
            raise ValueError("La barrière basse doit être inférieure à la barrière haute.")
        if not LOWER < S0 < UPPER:
            # Le produit est déjà knock-out si le spot est hors du corridor.
            return 0.0, 0.0
        if T <= 0:
            surv = 1.0 if LOWER < S0 < UPPER else 0.0
            return NOTIONAL * surv, surv
        if sigma < 0:
            raise ValueError("La volatilité doit être positive.")
        if N_PATHS <= 0:
            raise ValueError("Le nombre de trajectoires doit être positif.")

        N_STEPS = max(10, int(T * 365))
        dt = T / N_STEPS

        rng = np.random.default_rng(random_seed)
        z = rng.normal(size=(N_PATHS, N_STEPS))

        S = np.full(N_PATHS, S0, dtype=float)
        alive = np.ones(N_PATHS, dtype=bool)

        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        for i in range(N_STEPS):
            S *= np.exp(drift + diffusion * z[:, i])
            alive &= (S > LOWER) & (S < UPPER)

            # Petit gain de temps : si plus aucun chemin ne survit.
            if not alive.any():
                break

        surv = float(alive.mean())
        price = NOTIONAL * np.exp(-r * T) * surv
        return float(price), surv

    # ======================================================
    # Greeks par différences finies
    # ======================================================
    def compute_greeks(
        S0,
        LOWER,
        UPPER,
        sigma,
        r,
        T,
        NOTIONAL=10,
        N_PATHS=30000,
    ):
        # Common random numbers: même seed pour réduire le bruit MC
        seed = 42

        price, surv = dnt_mc(
            S0, LOWER, UPPER, sigma, r, T, NOTIONAL, N_PATHS, random_seed=seed
        )

        hs = max(1.0, 0.01 * float(S0))

        up = dnt_mc(
            S0 + hs, LOWER, UPPER, sigma, r, T, NOTIONAL, N_PATHS, random_seed=seed
        )[0]
        down = dnt_mc(
            S0 - hs, LOWER, UPPER, sigma, r, T, NOTIONAL, N_PATHS, random_seed=seed
        )[0]

        delta = (up - down) / (2 * hs)
        gamma = (up - 2 * price + down) / (hs**2)

        hv = 0.01
        up_vol = dnt_mc(
            S0,
            LOWER,
            UPPER,
            sigma + hv,
            r,
            T,
            NOTIONAL,
            N_PATHS,
            random_seed=seed,
        )[0]
        down_vol = dnt_mc(
            S0,
            LOWER,
            UPPER,
            max(0.01, sigma - hv),
            r,
            T,
            NOTIONAL,
            N_PATHS,
            random_seed=seed,
        )[0]

        vega = (up_vol - down_vol) / (2 * hv)
        # Vega pour +1 point de vol
        vega *= 0.01

        price_t1 = dnt_mc(
            S0,
            LOWER,
            UPPER,
            sigma,
            r,
            max(T - 1 / 365, 1e-6),
            NOTIONAL,
            N_PATHS,
            random_seed=seed,
        )[0]

        theta = price_t1 - price

        return price, surv, delta, gamma, vega, theta

    # ======================================================
    # PRICER MANUEL
    # ======================================================
    @reactive.calc
    @reactive.event(input.run)
    def calc():
        return compute_greeks(
            input.spot(),
            input.lower(),
            input.upper(),
            input.sigma(),
            input.rate(),
            input.days() / 365,
            input.notional(),
            int(input.npaths()),
        )

    @output
    @render.text
    def results():
        try:
            price, surv, delta, gamma, vega, theta = calc()
        except Exception as exc:
            return f"Erreur : {exc}"

        return f"""
Prix DNT      : {price:.4f}

Delta         : {delta:.6f}
Gamma         : {gamma:.6f}
Vega          : {vega:.6f}
Theta 1 jour  : {theta:.6f}

Survie        : {100 * surv:.2f} %
KO            : {100 * (1 - surv):.2f} %
"""

    # ======================================================
    # Graphique prix en fonction de la volatilité
    # ======================================================
    @output
    @render.plot
    @reactive.event(input.run)
    def vol_chart():
        sigma0 = float(input.sigma())
        vols = np.linspace(max(0.05, sigma0 - 0.15), sigma0 + 0.15, 25)

        prices = []
        for vol in vols:
            p, _ = dnt_mc(
                input.spot(),
                input.lower(),
                input.upper(),
                vol,
                input.rate(),
                input.days() / 365,
                input.notional(),
                min(int(input.npaths()), 30000),
                random_seed=123,
            )
            prices.append(p)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(vols, prices)
        ax.axvline(sigma0, linestyle="--", linewidth=1)
        ax.set_title("Prix DNT en fonction de la volatilité")
        ax.set_xlabel("Volatilité")
        ax.set_ylabel("Prix")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig

    # ======================================================
    # BATCH SG + RATIOS SHORT VOL / THETA CARRY
    # ======================================================
    @reactive.calc
    def batch_data():
        file = input.sg_file()

        if file is None:
            return pd.DataFrame()

        sigma = float(input.sigma())
        r = float(input.rate())
        nominal = float(input.notional())

        try:
            df = pd.read_excel(file[0]["datapath"])
        except Exception as exc:
            return pd.DataFrame({"Erreur": [f"Lecture Excel impossible : {exc}"]})

        results = []

        def parse_number(value):
            if pd.isna(value):
                raise ValueError("Valeur manquante")
            text = str(value).strip().replace("\u202f", " ").replace("\xa0", " ")
            text = text.split()[0]
            text = text.replace("€", "").replace("$", "")
            if "," in text and "." not in text:
                text = text.replace(",", ".")
            elif "," in text and "." in text:
                text = text.replace(",", "")
            return float(text)

        for _, row in df.iterrows():
            try:
                S0 = parse_number(row["Prix sous-jacent"])
                LOWER = parse_number(row["Barrière"])
                UPPER = parse_number(row["Borne haute"])

                maturity = pd.to_datetime(row["Maturité"], dayfirst=True)
                T = (maturity.normalize() - pd.Timestamp.today().normalize()).days / 365
                if T <= 0:
                    continue

                price, surv, delta, gamma, vega, theta = compute_greeks(
                    S0, LOWER, UPPER, sigma, r, T, nominal,
                    min(int(input.npaths()), 30000),
                )

                bid = parse_number(row["Achat"])
                ask = parse_number(row["Vente"])

                # Distance à la barrière la plus proche, en % du spot.
                dist_lower_pct = 100.0 * (S0 - LOWER) / S0
                dist_upper_pct = 100.0 * (UPPER - S0) / S0
                nearest_barrier_pct = min(dist_lower_pct, dist_upper_pct)

                abs_delta = abs(delta)

                # Ratios comparables entre warrants :
                # - Dist/|Delta| : grand = corridor éloigné pour peu d'exposition directionnelle.
                # - ShortVol/Prime : grand = Vega négatif important par euro de prime.
                # - Theta/Prime : grand = carry quotidien élevé par euro de prime.
                dist_delta_ratio = nearest_barrier_pct / max(abs_delta, 1e-8)
                short_vol_per_prime = (-vega) / ask if ask > 0 else np.nan
                theta_per_prime = theta / ask if ask > 0 else np.nan

                results.append({
                    "Mnemo": row.get("Mnémo.", row.get("Mnémo", "")),
                    "Price": round(price, 4),
                    "Maturity": maturity.date().isoformat(),
                    "Spot": round(S0, 4),
                    "Lower": round(LOWER, 4),
                    "Upper": round(UPPER, 4),
                    "Bid": bid,
                    "Ask": ask,
                    "Edge": round(price - ask, 4),
                    "Surv": round(surv, 4),
                    "DistBar%": round(nearest_barrier_pct, 3),
                    "AbsDelta": round(abs_delta, 6),
                    "Dist/|Delta|": round(dist_delta_ratio, 3),
                    "Delta": round(delta, 6),
                    "Gamma": round(gamma, 6),
                    "Vega": round(vega, 6),
                    "-Vega/Prime": round(short_vol_per_prime, 6) if np.isfinite(short_vol_per_prime) else np.nan,
                    "Theta": round(theta, 6),
                    "Theta/Prime": round(theta_per_prime, 6) if np.isfinite(theta_per_prime) else np.nan,
                })
            except Exception:
                continue

        if not results:
            return pd.DataFrame()

        out = pd.DataFrame(results)

        # Score de classement pur short vol, sans imposer d'unités arbitraires :
        # moyenne des rangs percentiles sur les 3 critères voulus.
        # 1 = meilleur : barrière loin, |delta| proche de 0, vega le plus négatif / prime.
        out["ShortVolScore"] = (
            out["DistBar%"].rank(pct=True, ascending=True)
            + out["AbsDelta"].rank(pct=True, ascending=False)
            + out["-Vega/Prime"].rank(pct=True, ascending=True)
        ) / 3.0
        out["ShortVolScore"] = out["ShortVolScore"].round(4)

        return out

    @output
    @render.data_frame
    def greeks_batch():
        out = batch_data()
        if out.empty:
            return render.DataGrid(out)
        if "Erreur" in out.columns:
            return render.DataGrid(out)
        return render.DataGrid(out.sort_values("Edge", ascending=False), filters=True)

    @output
    @render.data_frame
    def short_vol_batch():
        out = batch_data()
        if out.empty or "Erreur" in out.columns:
            return render.DataGrid(out)

        cols = [
            "Mnemo", "Maturity", "Ask", "Edge", "Surv", "DistBar%",
            "AbsDelta", "Dist/|Delta|", "Vega", "-Vega/Prime",
            "Theta", "Theta/Prime", "ShortVolScore"
        ]
        ranked = out.sort_values(
            ["ShortVolScore", "Dist/|Delta|", "-Vega/Prime"],
            ascending=[False, False, False],
        )[cols]
        return render.DataGrid(ranked, filters=True)

    @output
    @render.data_frame
    def theta_carry_batch():
        out = batch_data()
        if out.empty or "Erreur" in out.columns:
            return render.DataGrid(out)

        cols = [
            "Mnemo", "Maturity", "Ask", "Edge", "Surv", "DistBar%",
            "AbsDelta", "Vega", "Theta", "Theta/Prime", "ShortVolScore"
        ]
        ranked = out.sort_values("Theta/Prime", ascending=False)[cols]
        return render.DataGrid(ranked, filters=True)


app = App(app_ui, server)
