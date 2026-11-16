"""StabWarrant — Double-No-Touch NVDA avec import certificats SG Short.
Version autonome préparée à partir du fonctionnement du dépôt en date du 16/11/2026.
"""
from shiny import App, ui, render, reactive
import re
import unicodedata
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SG_SHORT_URL = (
    "https://sgbourse.fr/product-search/leverage%2520et%2520short/"
    "actions%2520am%C3%A9ricaines/nvidia/all-assets"
    "?ShowFilter=false&StrategyTypeAddition=Baissier"
)
SG_SHORT_EXPORT_URL = (
    "https://www.sgbourse.fr/EmcWebApi/api/ProductSearch/Export"
    "?PageNum=1&ProductClassificationId=5515"
    "&AssetTypeId=1&AssetTypeMenuId=33&AssetId=320"
)
SHORT_COLS = ["Mnémo", "ISIN", "Type", "Sous-jacent", "Levier", "Bid_EUR",
              "Ask_EUR", "Spot_USD", "Frais_gestion_pct", "Source", "Date_extraction_UTC"]


def _number(value):
    if value is None or pd.isna(value):
        return np.nan
    txt = str(value).replace("−", "-").replace("–", "-")
    txt = txt.replace("\u202f", "").replace("\xa0", "").replace(" ", "")
    m = re.search(r"[-+]?\d[\d.,]*", txt)
    if not m:
        return np.nan
    txt = m.group()
    if "," in txt:
        txt = txt.replace(".", "").replace(",", ".")
    try:
        return float(txt)
    except ValueError:
        return np.nan


def import_shorts(path, filename):
    """Lit un export SG Excel/CSV. Exclut les produits non-Short et non-NVIDIA."""
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    elif filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype=str)
    else:
        raise ValueError("Formats autorisés : CSV, XLS ou XLSX.")

    def norm(v):
        v = unicodedata.normalize("NFKD", str(v))
        return " ".join("".join(c for c in v if not unicodedata.combining(c)).lower().split())

    aliases = {
        "mnemo": "Mnémo", "mnemo.": "Mnémo", "code": "Mnémo", "code produit": "Mnémo",
        "isin": "ISIN", "code isin": "ISIN", "type": "Type", "type de produit": "Type",
        "produit": "Type", "strategie": "Type", "sens": "Type",
        "sous-jacent": "Sous-jacent", "sous jacent": "Sous-jacent",
        "nom de l'actif": "Sous-jacent", "actif": "Sous-jacent",
        "levier": "Levier", "levier quotidien": "Levier", "levier actuel": "Levier",
        "effet de levier": "Levier", "leverage": "Levier", "levier indicatif": "Levier",
        "achat": "Bid_EUR", "bid": "Bid_EUR", "bid_eur": "Bid_EUR",
        "vente": "Ask_EUR", "ask": "Ask_EUR", "ask_eur": "Ask_EUR",
        "prix sous-jacent": "Spot_USD", "cours sous-jacent": "Spot_USD",
        "spot_usd": "Spot_USD", "frais de gestion (%)": "Frais_gestion_pct",
        "frais de gestion": "Frais_gestion_pct",
    }
    df = df.rename(columns={c: aliases.get(norm(c), str(c).strip()) for c in df.columns})
    df = df.loc[:, ~df.columns.duplicated()].copy()
    required = {"Mnémo", "Levier", "Ask_EUR"}
    if not required.issubset(df.columns):
        raise ValueError("Colonnes Short manquantes : " + ", ".join(sorted(required - set(df.columns))))

    rows = []
    for _, row in df.iterrows():
        underlying = str(row.get("Sous-jacent", "")).strip().upper()
        if underlying not in ("", "NAN", "NONE", "-") and not any(
            x in underlying for x in ("NVIDIA", "NVDA")
        ):
            continue
        kind = str(row.get("Type", "")).lower()
        leverage = _number(row.get("Levier"))
        if np.isfinite(leverage) and leverage > 0 and (
            "short" in kind or "baissier" in kind
        ):
            leverage = -leverage
        if not np.isfinite(leverage) or leverage >= 0:
            continue
        ask = _number(row.get("Ask_EUR"))
        memo = str(row.get("Mnémo", "")).strip()
        if not np.isfinite(ask) or ask <= 0 or memo.lower() in ("", "nan", "none"):
            continue
        rows.append({
            "Mnémo": memo, "ISIN": row.get("ISIN"), "Type": "Short",
            "Sous-jacent": "NVIDIA", "Levier": leverage,
            "Bid_EUR": _number(row.get("Bid_EUR")), "Ask_EUR": ask,
            "Spot_USD": _number(row.get("Spot_USD")),
            "Frais_gestion_pct": _number(row.get("Frais_gestion_pct")),
        })
    if not rows:
        raise ValueError("Aucun Short NVIDIA valide : vérifier Mnémo, Levier négatif et Vente.")
    out = pd.DataFrame(rows)
    out["Source"] = f"Import utilisateur : {filename}"
    out["Date_extraction_UTC"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out[SHORT_COLS].drop_duplicates(subset=["Mnémo", "ISIN"]).reset_index(drop=True)


def build_delta_hedges(stabilities, shorts, stability_qty=1):
    """Estimation de hedge LONG Stability delta positif avec Short SG.

    Delta Short (EUR/USD) ~= levier quotidien * Ask(EUR) / Spot(USD).
    Il s'agit d'une approximation instantanée, non d'un hedge permanent.
    """
    cols = ["Stability", "Spot_USD", "Delta_stability_EUR_USD", "Short",
            "Levier_short", "Bid_short_EUR", "Ask_short_EUR", "Delta_short_est_EUR_USD",
            "Spread_short_pct", "Quantite_ideale", "Quantite_entiere", "Cout_hedge_EUR",
            "Delta_residuel_EUR_USD", "Couverture_pct", "Statut"]
    if stabilities.empty or shorts.empty:
        return pd.DataFrame(columns=cols)
    w = shorts.copy()
    for c in ("Levier", "Bid_EUR", "Ask_EUR", "Spot_USD"):
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w[(w["Levier"] < 0) & (w["Ask_EUR"] > 0)].copy()
    rows = []
    for _, stab in stabilities.iterrows():
        try:
            delta = float(stab["Delta"]) * int(stability_qty)
            spot = float(stab["Spot"])
            if not np.isfinite(delta) or not np.isfinite(spot) or spot <= 0:
                continue
            base = {"Stability": stab["Mnemo"], "Spot_USD": round(spot, 4),
                    "Delta_stability_EUR_USD": round(delta, 7)}
            if delta <= 1e-8:
                rows.append({**base, "Statut": "Delta nul ou négatif : Short inadapté"})
                continue
            candidates = w.copy()
            candidates["Spot_ref"] = candidates["Spot_USD"].fillna(spot)
            candidates = candidates[(candidates["Spot_ref"] > 0) &
                                    ((candidates["Spot_ref"] - spot).abs() / spot <= 0.03)].copy()
            if candidates.empty:
                rows.append({**base, "Statut": "Aucun Short avec spot compatible (3 %)"})
                continue
            candidates["Delta_short"] = (candidates["Levier"] * candidates["Ask_EUR"] /
                                         candidates["Spot_ref"])
            candidates = candidates[np.isfinite(candidates["Delta_short"]) &
                                    (candidates["Delta_short"] < -1e-9)].copy()
            if candidates.empty:
                rows.append({**base, "Statut": "Delta Short non calculable"})
                continue
            candidates["Spread_pct"] = np.where(
                (candidates["Bid_EUR"] > 0) & (candidates["Bid_EUR"] <= candidates["Ask_EUR"]),
                100 * (candidates["Ask_EUR"] - candidates["Bid_EUR"]) / candidates["Ask_EUR"],
                np.inf,
            )
            candidates["AbsLevier"] = candidates["Levier"].abs()
            best = candidates.sort_values(["Spread_pct", "AbsLevier", "Ask_EUR"]).iloc[0]
            delta_short = float(best["Delta_short"])
            ideal = -delta / delta_short
            qty = max(0, int(np.floor(ideal + 0.5)))
            residual = delta + qty * delta_short
            rows.append({
                **base, "Short": best["Mnémo"], "Levier_short": float(best["Levier"]),
                "Bid_short_EUR": round(float(best["Bid_EUR"]), 4)
                if np.isfinite(best["Bid_EUR"]) else np.nan,
                "Ask_short_EUR": round(float(best["Ask_EUR"]), 4),
                "Delta_short_est_EUR_USD": round(delta_short, 8),
                "Spread_short_pct": round(float(best["Spread_pct"]), 3)
                if np.isfinite(best["Spread_pct"]) else np.nan,
                "Quantite_ideale": round(ideal, 3), "Quantite_entiere": qty,
                "Cout_hedge_EUR": round(qty * float(best["Ask_EUR"]), 2),
                "Delta_residuel_EUR_USD": round(residual, 7),
                "Couverture_pct": round(100 * (1 - abs(residual) / abs(delta)), 2),
                "Statut": "Estimation instantanée : actualiser les cours et le levier",
            })
        except (KeyError, ValueError, TypeError, OverflowError, ZeroDivisionError):
            continue
    return pd.DataFrame(rows, columns=cols)


# ==============================================================
# Double-No-Touch Monte Carlo (modèle GBM, barrières discrètes)
# ==============================================================
def dnt_mc(S0, lower, upper, sigma, r, T, notional=10, npaths=30000, seed=42):
    S0, lower, upper, sigma, T = map(float, (S0, lower, upper, sigma, T))
    if not lower < upper:
        raise ValueError("La barrière basse doit être inférieure à la barrière haute")
    if not lower < S0 < upper:
        return 0.0, 0.0
    if T <= 0:
        return float(notional), 1.0
    if sigma < 0 or npaths <= 0:
        raise ValueError("Volatilité ou nombre de simulations invalide")
    steps = max(10, int(T * 365))
    dt = T / steps
    rng = np.random.default_rng(seed)
    spot = np.full(int(npaths), S0)
    alive = np.ones(int(npaths), dtype=bool)
    drift, diffusion = (r - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt)
    # Générer les chocs par pas évite de conserver un énorme tableau en RAM.
    for _ in range(steps):
        spot *= np.exp(drift + diffusion * rng.standard_normal(int(npaths)))
        alive &= (spot > lower) & (spot < upper)
        if not alive.any():
            break
    surv = float(alive.mean())
    return float(notional * np.exp(-r * T) * surv), surv


def compute_greeks(S0, lower, upper, sigma, r, T, notional=10, npaths=30000):
    npaths = int(npaths)
    price, surv = dnt_mc(S0, lower, upper, sigma, r, T, notional, npaths)
    hs = max(1, 0.01 * float(S0))
    up = dnt_mc(S0 + hs, lower, upper, sigma, r, T, notional, npaths)[0]
    down = dnt_mc(S0 - hs, lower, upper, sigma, r, T, notional, npaths)[0]
    delta = (up - down) / (2 * hs)
    gamma = (up - 2 * price + down) / hs**2
    hv = 0.01
    upv = dnt_mc(S0, lower, upper, sigma + hv, r, T, notional, npaths)[0]
    downv = dnt_mc(S0, lower, upper, max(0.01, sigma - hv), r, T, notional, npaths)[0]
    vega = (upv - downv) / (2 * hv) * 0.01  # 1 point de volatilité
    nextday = dnt_mc(S0, lower, upper, sigma, r, max(T - 1/365, 1e-6), notional, npaths)[0]
    theta = nextday - price
    return price, surv, delta, gamma, vega, theta


def _number_sg(value):
    val = _number(value)
    if not np.isfinite(val):
        raise ValueError(f"Valeur manquante ou invalide : {value}")
    return val


STAB_EXPORT_URL = (
    "https://www.sgbourse.fr/EmcWebApi/api/ProductSearch/Export"
    "?PageNum=1&ProductClassificationId=8&AssetId=320"
)


def _export_excel(url, referer=None):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
                  "application/vnd.ms-excel,application/octet-stream,*/*",
    }
    if referer:
        headers["Referer"] = referer
    with urlopen(Request(url, headers=headers), timeout=40) as response:
        first = response.read(8192)
        if not (first.startswith(b"PK\x03\x04") or first.startswith(b"\xd0\xcf\x11\xe0")):
            raise ValueError("SG n'a pas renvoyé un fichier Excel valide")
        yield first
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            yield chunk


def portfolio_greeks(batch, positions):
    """Aggregate per-unit greeks from Batch SG against portfolio positions.

    Theta is the app's 1-day finite-difference theta, Vega per 1 point of vol,
    and Delta per USD move. No live SG quote or hedge is implied.
    """
    cols = ["TICKER", "NB STAB", "DELTA", "DELTA ptf", "VEGA", "Vega ptf",
            "THETA", "Theta ptf"]
    if batch.empty or "Erreur" in batch.columns:
        return pd.DataFrame(columns=cols), ["Importer d'abord l'export Stability SG (Batch SG)."]
    required = {"Mnemo", "Delta", "Vega", "Theta"}
    if not required.issubset(batch.columns):
        return pd.DataFrame(columns=cols), ["Colonnes de grecs absentes du Batch SG."]
    base = batch.copy()
    base["_ticker"] = base["Mnemo"].astype(str).str.strip().str.upper()
    base = base.drop_duplicates("_ticker", keep="first").set_index("_ticker")
    result, messages = [], []
    grouped = {}
    for ticker, quantity in positions:
        ticker = str(ticker).strip().upper()
        if not ticker:
            continue
        try:
            quantity = int(quantity)
        except (TypeError, ValueError):
            messages.append(f"Quantité non valide pour {ticker}.")
            continue
        if quantity == 0:
            continue
        grouped[ticker] = grouped.get(ticker, 0) + quantity
    for ticker, quantity in grouped.items():
        if ticker not in base.index:
            messages.append(f"{ticker} : absent du Batch SG importé (mnémo introuvable ou produit expiré).")
            continue
        row = base.loc[ticker]
        values = pd.to_numeric(row[["Delta", "Vega", "Theta"]], errors="coerce")
        if values.isna().any():
            messages.append(f"{ticker} : grecs indisponibles.")
            continue
        delta, vega, theta = [float(x) for x in values]
        result.append({
            "TICKER": ticker, "NB STAB": quantity,
            "DELTA": delta, "DELTA ptf": quantity * delta,
            "VEGA": vega, "Vega ptf": quantity * vega,
            "THETA": theta, "Theta ptf": quantity * theta,
        })
    return pd.DataFrame(result, columns=cols), messages


app_ui = ui.page_fluid(
    ui.h2("Stability / Double No Touch Pricer — NVIDIA"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_numeric("spot", "Spot NVIDIA USD", 162.716),
            ui.input_numeric("lower", "Barrière basse", 140),
            ui.input_numeric("upper", "Barrière haute", 200),
            ui.input_numeric("sigma", "Volatilité (ex. 0.50)", 0.50),
            ui.input_numeric("rate", "Taux", 0.04),
            ui.input_numeric("days", "Jours jusqu'à maturité", 90),
            ui.input_numeric("notional", "Nominal du Stability €", 10),
            ui.input_numeric("npaths", "Nombre de trajectoires", 30000, min=1000),
            ui.download_button("download_sg", "Télécharger export Stability SG"),
            ui.input_file("sg_file", "Importer Stability SG Excel", accept=[".xlsx", ".xls"]),
            ui.download_button("download_shorts_sg", "Télécharger Short SG NVIDIA Excel"),
            ui.input_file("shorts_file", "Importer Short NVIDIA Excel/CSV", accept=[".xlsx", ".xls", ".csv"]),
            ui.download_button("export_shorts", "Exporter Short CSV"),
            ui.input_numeric("stability_qty", "Nombre de Stability détenus", 75, min=1, step=1),
            ui.download_button("download_hedges", "Exporter couvertures Delta CSV"),
            ui.input_action_button("run", "Calculer"),
            width=330,
        ),
        ui.navset_tab(
            ui.nav_panel("Pricer", ui.output_text_verbatim("results"), ui.output_plot("vol_chart")),
            ui.nav_panel("Batch SG", ui.output_data_frame("greeks_batch")),
            ui.nav_panel(
                "Portefeuille",
                ui.h4("Portefeuille Stability NVIDIA"),
                ui.p("Saisissez chaque mnémonique et sa quantité. Les grecs unitaires sont "
                     "récupérés depuis l'export Stability SG importé dans la barre latérale."),
                ui.p("Quantités positives = positions longues ; négatives = positions courtes."),
                *[
                    ui.row(
                        ui.column(6, ui.input_text(
                            f"ptf_ticker_{i}", f"Ticker {i + 1}",
                            placeholder="Ex. 80N4S" if i == 0 else "Mnémonique SG",
                        )),
                        ui.column(6, ui.input_numeric(
                            f"ptf_qty_{i}", f"Quantité {i + 1}", 0, step=1,
                        )),
                    ) for i in range(10)
                ],
                ui.hr(),
                ui.output_text("portfolio_status"),
                ui.output_data_frame("portfolio_table"),
                ui.h4("Grecs totaux du portefeuille"),
                ui.output_text_verbatim("portfolio_totals"),
                ui.download_button("download_portfolio", "Exporter le portefeuille (CSV Excel)"),
                ui.p("Delta : €/USD ; Vega : € par point de vol ; "
                     "Theta : € par jour. Tous sont calculés à partir du même modèle Batch SG."),
            ),
            ui.nav_panel("Pur Short Vol", ui.output_data_frame("short_vol_batch")),
            ui.nav_panel(
                "Delta Hedge",
                ui.p("Import des certificats NVIDIA Short SG ; delta instantané approximatif = "
                     "levier quotidien × Ask / spot NVIDIA."),
                ui.p("Source : ", ui.a("SG Bourse — certificats Short NVIDIA", href=SG_SHORT_URL, target="_blank")),
                ui.output_text("shorts_status"),
                ui.output_data_frame("shorts_table"),
                ui.output_text("hedge_status"),
                ui.output_data_frame("delta_hedge_table"),
            ),
            ui.nav_panel("Theta Carry", ui.output_data_frame("theta_carry_batch")),
        ),
    ),
)


def server(input, output, session):
    @output
    @render.download_button(filename="sg_stability_export.xlsx")
    def download_sg():
        yield from _export_excel(STAB_EXPORT_URL)

    @output
    @render.download_button(filename="sg_nvidia_shorts.xlsx")
    def download_shorts_sg():
        yield from _export_excel(SG_SHORT_EXPORT_URL, SG_SHORT_URL)

    @reactive.calc
    def shorts_data():
        files = input.shorts_file()
        if not files:
            return pd.DataFrame(columns=SHORT_COLS), "Non chargé", None
        item = files[0]
        try:
            return import_shorts(item["datapath"], item["name"]), "Import utilisateur", None
        except Exception as exc:
            return pd.DataFrame(columns=SHORT_COLS), "Import impossible", str(exc)

    @output
    @render.text
    def shorts_status():
        data, source, error = shorts_data()
        if error:
            return f"{source}: {error}"
        if data.empty:
            return "Importer un export SG Short NVIDIA pour calculer le hedge."
        return f"{len(data)} certificats Short chargés ({source})"

    @output
    @render.data_frame
    def shorts_table():
        return render.DataGrid(shorts_data()[0], filters=True)

    @output
    @render.download_button(filename=lambda: f"sg_nvidia_shorts_{datetime.now():%Y%m%d_%H%M}.csv")
    def export_shorts():
        df, _, err = shorts_data()
        if err or df.empty:
            raise ValueError(err or "Aucun Short chargé")
        yield df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")

    @reactive.calc
    @reactive.event(input.run)
    def manual_calc():
        return compute_greeks(
            input.spot(), input.lower(), input.upper(), input.sigma(),
            input.rate(), input.days() / 365, input.notional(), min(int(input.npaths()), 30000)
        )

    @output
    @render.text
    def results():
        try:
            price, surv, delta, gamma, vega, theta = manual_calc()
            return (f"Prix DNT : {price:.4f} €\n"
                    f"Survie : {100*surv:.2f}%\n"
                    f"Delta : {delta:.6f} €/USD\n"
                    f"Gamma : {gamma:.6f}\n"
                    f"Vega (+1 pt vol) : {vega:.6f} €/pt\n"
                    f"Theta (1 jour) : {theta:.6f} €/jour")
        except Exception as exc:
            return f"Erreur de pricing : {exc}"

    @output
    @render.plot
    def vol_chart():
        x = np.linspace(max(0.05, input.sigma() - 0.15), input.sigma() + 0.15, 11)
        vals = [dnt_mc(input.spot(), input.lower(), input.upper(), v, input.rate(),
                       input.days()/365, input.notional(), min(5000, int(input.npaths())))[0] for v in x]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x * 100, vals, marker="o")
        ax.set_xlabel("Volatilité annuelle (%)")
        ax.set_ylabel("Prix DNT (€)")
        ax.set_title("Sensibilité théorique à la volatilité")
        fig.tight_layout()
        return fig

    @reactive.calc
    def batch_data():
        files = input.sg_file()
        if not files:
            return pd.DataFrame()
        try:
            df = pd.read_excel(files[0]["datapath"])
        except Exception as exc:
            return pd.DataFrame({"Erreur": [f"Import Excel impossible : {exc}"]})
        rows = []
        for _, row in df.iterrows():
            try:
                S0 = _number_sg(row["Prix sous-jacent"])
                lower = _number_sg(row["Barrière"])
                upper = _number_sg(row["Borne haute"])
                expiry = pd.to_datetime(row["Maturité"], dayfirst=True)
                T = (expiry.normalize() - pd.Timestamp.today().normalize()).days / 365
                if T <= 0:
                    continue
                price, surv, delta, gamma, vega, theta = compute_greeks(
                    S0, lower, upper, input.sigma(), input.rate(), T,
                    input.notional(), min(int(input.npaths()), 30000)
                )
                bid = _number_sg(row["Achat"])
                ask = _number_sg(row["Vente"])
                distance = 100 * min((S0-lower)/S0, (upper-S0)/S0)
                rows.append({
                    "Mnemo": str(row.get("Mnémo", row.get("Mnémo.", ""))),
                    "Maturity": expiry.strftime("%Y-%m-%d"),
                    "Spot": S0, "Lower": lower, "Upper": upper,
                    "Bid": bid, "Ask": ask, "Price": price,
                    "Edge": round(price - ask, 4), "Surv": round(surv, 4),
                    "DistBar%": round(distance, 3),
                    "AbsDelta": abs(delta), "Delta": delta, "Gamma": gamma,
                    "Vega": vega, "Theta": theta,
                    "Dist/|Delta|": distance/max(abs(delta), 1e-8),
                    "-Vega/Prime": -vega/ask if ask > 0 else np.nan,
                    "Theta/Prime": theta/ask if ask > 0 else np.nan,
                })
            except (ValueError, KeyError, TypeError, OverflowError):
                continue
        out = pd.DataFrame(rows)
        if not out.empty:
            out["ShortVolScore"] = (
                out["DistBar%"].rank(pct=True, ascending=True)
                + out["AbsDelta"].rank(pct=True, ascending=False)
                + out["-Vega/Prime"].rank(pct=True, ascending=True)
            ) / 3
        return out

    @output
    @render.data_frame
    def greeks_batch():
        return render.DataGrid(batch_data(), filters=True)

    @reactive.calc
    def portfolio_data():
        positions = [
            (getattr(input, f"ptf_ticker_{i}")(), getattr(input, f"ptf_qty_{i}")())
            for i in range(10)
        ]
        return portfolio_greeks(batch_data(), positions)

    @output
    @render.text
    def portfolio_status():
        data, warnings = portfolio_data()
        message = f"{len(data)} ticker(s) calculé(s)."
        if warnings:
            message += " Attention : " + " | ".join(warnings)
        return message

    @output
    @render.data_frame
    def portfolio_table():
        data, _ = portfolio_data()
        if data.empty:
            return render.DataGrid(data)
        numeric = ["DELTA", "DELTA ptf", "VEGA", "Vega ptf", "THETA", "Theta ptf"]
        view = data.copy()
        view[numeric] = view[numeric].round(6)
        totals = {"TICKER": "TOTAL", "NB STAB": int(data["NB STAB"].sum()),
                  "DELTA": np.nan, "VEGA": np.nan, "THETA": np.nan,
                  "DELTA ptf": data["DELTA ptf"].sum(),
                  "Vega ptf": data["Vega ptf"].sum(),
                  "Theta ptf": data["Theta ptf"].sum()}
        return render.DataGrid(pd.concat([view, pd.DataFrame([totals])], ignore_index=True), filters=False)

    @output
    @render.text
    def portfolio_totals():
        data, warnings = portfolio_data()
        if data.empty:
            return "Aucune position calculée. Importez l'Excel SG puis entrez un ticker et sa quantité."
        return (f"DELTA PTF : {data['DELTA ptf'].sum():+.6f} €/USD\n"
                f"VEGA PTF  : {data['Vega ptf'].sum():+.6f} €/point de vol\n"
                f"THETA PTF : {data['Theta ptf'].sum():+.6f} €/jour")

    @output
    @render.download_button(filename="stability_portefeuille_greeks.csv")
    def download_portfolio():
        data, warnings = portfolio_data()
        if data.empty:
            raise ValueError("Aucune position valide : importer SG et saisir les tickers/quantités")
        if warnings:
            raise ValueError("Export bloqué : corriger les positions non calculées. "
                             + " | ".join(warnings))
        totals = {"TICKER": "TOTAL", "NB STAB": int(data["NB STAB"].sum()),
                  "DELTA": np.nan, "VEGA": np.nan, "THETA": np.nan,
                  "DELTA ptf": data["DELTA ptf"].sum(),
                  "Vega ptf": data["Vega ptf"].sum(),
                  "Theta ptf": data["Theta ptf"].sum()}
        result = pd.concat([data, pd.DataFrame([totals])], ignore_index=True)
        yield result.to_csv(index=False, sep=";", decimal=",", encoding="utf-8-sig").encode("utf-8-sig")

    @output
    @render.data_frame
    def short_vol_batch():
        d = batch_data()
        if d.empty or "Erreur" in d:
            return render.DataGrid(d)
        return render.DataGrid(d.sort_values("ShortVolScore", ascending=False), filters=True)

    @output
    @render.data_frame
    def theta_carry_batch():
        d = batch_data()
        if d.empty or "Erreur" in d:
            return render.DataGrid(d)
        return render.DataGrid(d.sort_values("Theta/Prime", ascending=False), filters=True)

    @reactive.calc
    def delta_hedges():
        d = batch_data()
        shorts, _, err = shorts_data()
        if err or d.empty or "Erreur" in d:
            return pd.DataFrame()
        return build_delta_hedges(d, shorts, input.stability_qty())

    @output
    @render.text
    def hedge_status():
        if input.sg_file() is None:
            return "Importer d'abord les Stability SG."
        shorts, _, err = shorts_data()
        if err:
            return f"Erreur Short : {err}"
        if shorts.empty:
            return "Importer ensuite le fichier Short SG NVIDIA."
        result = delta_hedges()
        return f"{result['Short'].notna().sum() if not result.empty else 0} hedges Short proposés."

    @output
    @render.data_frame
    def delta_hedge_table():
        return render.DataGrid(delta_hedges(), filters=True)

    @output
    @render.download_button(filename="nvda_stability_short_hedges.csv")
    def download_hedges():
        df = delta_hedges()
        if df.empty:
            raise ValueError("Importer les exports SG Stability et Short avant l'export")
        yield df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


app = App(app_ui, server)
