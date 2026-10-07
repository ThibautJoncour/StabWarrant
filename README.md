# Stability / Double No Touch Pricer — Shiny for Python

Projet reconstruit à partir des captures fournies.

## Fonctionnalités

- Pricer Monte Carlo d'un **Double No Touch** / Stability Warrant.
- Entrées : spot, barrières basse/haute, volatilité, taux, maturité, nominal, nombre de trajectoires.
- Calcul des Greeks par différences finies : **Delta, Gamma, Vega, Theta 1 jour**.
- Probabilité de survie et probabilité de KO.
- Courbe du prix en fonction de la volatilité.
- Import Excel d'un export Société Générale et calcul en batch.
- Classement des produits par `Edge = Price - Ask`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\\Scripts\\activate     # Windows
pip install -r requirements.txt
```

## Lancer l'application

```bash
shiny run --reload app.py
```

Puis ouvrir l'adresse indiquée dans le terminal (souvent `http://127.0.0.1:8000`).

## Colonnes attendues dans l'Excel SG

Le batch cherche les colonnes suivantes :

- `Prix sous-jacent`
- `Barrière`
- `Borne haute`
- `Maturité`
- `Achat`
- `Vente`
- `Mnémo.` ou `Mnémo`

Les valeurs texte au format français avec virgule décimale sont acceptées.

## Remarque modèle

Le sous-jacent est simulé sous une dynamique GBM risque-neutre :

`dS/S = r dt + sigma dW`

Le payoff vaut le nominal uniquement si aucune des deux barrières n'est touchée pendant toute la vie du produit.
