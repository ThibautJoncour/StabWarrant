from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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
        ),
    ),
)


# ==========================================================
# SERVER
# ==========================================================
def server(input, output, session):

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
    # BATCH SG
    # ======================================================
    @output
    @render.data_frame
    def greeks_batch():
        file = input.sg_file()

        if file is None:
            return render.DataGrid(pd.DataFrame())

        sigma = float(input.sigma())
        r = float(input.rate())
        nominal = float(input.notional())

        try:
            df = pd.read_excel(file[0]["datapath"])
        except Exception as exc:
            return render.DataGrid(pd.DataFrame({"Erreur": [f"Lecture Excel impossible : {exc}"]}))

        results = []

        def parse_number(value):
            if pd.isna(value):
                raise ValueError("Valeur manquante")
            text = str(value).strip().replace("\u202f", " ").replace("\xa0", " ")
            # Conserve la première partie avant un éventuel symbole / texte.
            text = text.split()[0]
            text = text.replace("€", "").replace("$", "")
            # Format français : 225,51
            if "," in text and "." not in text:
                text = text.replace(",", ".")
            elif "," in text and "." in text:
                # suppose séparateur de milliers = espace/virgule, décimale = point
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
                    S0,
                    LOWER,
                    UPPER,
                    sigma,
                    r,
                    T,
                    nominal,
                    min(int(input.npaths()), 30000),
                )

                bid = parse_number(row["Achat"])
                ask = parse_number(row["Vente"])

                results.append(
                    {
                        "Mnemo": row.get("Mnémo.", row.get("Mnémo", "")),
                        "Price": round(price, 4),
                        "Maturity": maturity.date().isoformat(),
                        "Bid": bid,
                        "Ask": ask,
                        "Edge": round(price - ask, 4),
                        "Surv": round(surv, 4),
                        "Delta": round(delta, 4),
                        "Gamma": round(gamma, 4),
                        "Vega": round(vega, 4),
                        "Theta": round(theta, 4),
                    }
                )
            except Exception:
                continue

        if not results:
            return render.DataGrid(pd.DataFrame())

        out = pd.DataFrame(results).sort_values("Edge", ascending=False)
        return render.DataGrid(out, filters=True)


app = App(app_ui, server)
