from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from urllib.request import Request, urlopen

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
    @render.download(filename="sg_stability_export.xlsx")
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
