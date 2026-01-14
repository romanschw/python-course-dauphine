# Session 2 : Composition de Classes et Chargement de Données

## Objectifs de la session

1. Introduire la programmation orientée-objet et les principes fondamentaux de conception de classes
2. Utiliser les types de collections intégrés à Python (listes, dictionnaires, ensembles)
3. Maîtriser la syntaxe des expressions de compréhension
4. Récupérer des données à partir d'APIs publiques
5. Implémenter un système de cache avec gestion des chevauchements temporels

---

## Contexte du projet

Dans la Session 1, nous avons créé notre classe `PriceSeries` pour représenter et manipuler des séries de prix. Nous pouvons maintenant construire des abstractions de plus haut niveau :

1. Une classe **Asset** qui *contient* une `PriceSeries` (composition)
2. Une classe **DataLoader** pour récupérer des données de marché avec cache intelligent
3. Une classe **Universe** pour gérer une collection d'actifs

### Architecture cible

```
┌─────────────────────────────────────────────────────────────┐
│                        Universe                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  Asset  │  │  Asset  │  │  Asset  │  │  Asset  │  ...   │
│  │  AAPL   │  │  MSFT   │  │  GOOGL  │  │  AMZN   │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │              │
│  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐        │
│  │  Price  │  │  Price  │  │  Price  │  │  Price  │        │
│  │  Series │  │  Series │  │  Series │  │  Series │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ fetch
                    ┌───────┴───────┐
                    │  DataLoader   │
                    │  (+ cache)    │
                    └───────────────┘
```

### Exemple d'utilisation finale

```python
from pyvest.core.priceseries import PriceSeries
from pyvest.core.asset import Asset
from pyvest.data.loader import DataLoader
from pyvest.core.universe import Universe

# Instanciation du DataLoader avec système de cache
loader = DataLoader(cache_dir=".cache")

# Récupération des données pour un actif via l'API Yahoo Finance
apple_ts = loader.fetch_single_ticker("AAPL", "Close", ("2024-01-01", "2024-12-01"))

# Test du système de cache (le second appel devrait être instantané)
apple_ts = loader.fetch_single_ticker("AAPL", "Close", ("2024-01-01", "2024-12-01"))

# Création de nos objets Asset
apple = Asset("AAPL", apple_ts, sector="Technology")
msft = Asset("MSFT", loader.fetch_single_ticker("MSFT", "Close", ("2024-01-01", "2024-12-01")), sector="Technology")

# Communication entre Asset et son interface PriceSeries
print(f"Volatilité AAPL: {apple.volatility:.2%}")

# Corrélation entre deux actifs
correlation = apple.correlation_with(msft)
print(f"Corrélation AAPL-MSFT: {correlation:.2f}")

# Agrégation dans un objet Universe
universe = Universe([apple, msft])
for asset in universe:
    print(f"{asset.ticker}: {asset.total_return:.2%}")
```

---

## Partie 1 : Développement Guidé

---

### Étape 2.1 : La classe Asset et le pattern Composition

#### Introduction à la Programmation Orientée Objet

La **programmation orientée-objet (POO)** est un paradigme qui permet de structurer le code autour d'**objets** — des entités définies par leurs caractéristiques (attributs) et leurs comportements (méthodes). Une **classe** agit comme un patron (ou usine) capable de produire des objets d'un type donné.

**Les trois piliers de la POO :**

| Pilier | Description | Exemple |
|--------|-------------|---------|
| **Encapsulation** | Regrouper données et comportements dans une même unité | `Asset` encapsule ticker, prix, secteur |
| **Abstraction** | Exposer une interface simple, cacher la complexité | `asset.volatility` cache le calcul sous-jacent |
| **Polymorphisme** | Objets de types différents répondant à la même interface | Tout objet avec `__len__` fonctionne avec `len()` |

> **Note importante** : La POO n'est pas toujours nécessaire. Un programme simple peut fonctionner plus clairement comme une combinaison de fonctions. Voir la présentation de Jack Diederich "Stop Writing Classes" (PyCon 2012). Cependant, pour une librairie structurée comme la nôtre, la POO apporte une organisation claire et maintenable.

#### Analyse des besoins

Avant de coder, identifions les classes principales et leurs responsabilités :

| Classe | Responsabilité | Attributs | Méthodes |
|--------|----------------|-----------|----------|
| **PriceSeries** | Stocke et opère sur une série de prix | `values`, `name` | `log_return()`, `volatility()` |
| **Asset** | Représente un actif financier et ses métadonnées | `ticker`, `prices`, `sector` | `correlation_with()`, propriétés métriques |
| **DataLoader** | Récupère et met en cache les données de marché | `cache_dir` | `fetch_single_ticker()`, `fetch_multiple_tickers()` |
| **Universe** | Gère une collection d'actifs | `_assets` | `add()`, `get()`, `filter_by_sector()` |

#### Les principes SOLID

Ces principes guident la conception de classes robustes et maintenables :

| Principe | Description | Application |
|----------|-------------|-------------|
| **S**ingle Responsibility | Une classe = une responsabilité claire | `DataLoader` ne fait que charger des données |
| **O**pen/Closed | Extensible sans modification interne | Héritage pour spécialiser `Asset` |
| **L**iskov Substitution | Sous-classes substituables aux parents | Polymorphisme |
| **I**nterface Segregation | Interfaces minimales et ciblées | Méthodes publiques essentielles uniquement |
| **D**ependency Inversion | Dépendre d'abstractions, pas d'implémentations | Injection de dépendances |

#### Types de relations entre classes

| Relation | Question à se poser | Exemple |
|----------|---------------------|---------|
| **Composition** | "B appartient-il à A ? Le cycle de vie de B dépend-il de A ?" | `Asset` *contient* une `PriceSeries` |
| **Agrégation** | "A contient-il B alors que B peut exister indépendamment ?" | `Universe` *contient* des `Asset` |
| **Héritage** | "A est-il un type spécialisé de B ?" | Non utilisé pour l'instant |

---

#### Implémentation de la classe Asset

```python
# Fichier: pyvest/core/asset.py

from pyvest.core.priceseries import PriceSeries
from enum import Enum


class CurrencyEnum(Enum):
    """Énumération des devises supportées."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"


class Asset:
    """
    Représente un actif financier avec son historique de prix.
    
    Pattern de conception : COMPOSITION
    ───────────────────────────────────
    Asset POSSÈDE une PriceSeries (relation HAS-A, pas IS-A).
    
    D'après 'Python OOP' (Lott & Phillips):
    "La composition est généralement le bon choix quand un objet 
    fait partie d'un autre objet."
    
    Le cycle de vie de PriceSeries est lié à celui de Asset :
    - Créé quand Asset est créé
    - Détruit quand Asset est détruit
    
    Attributes:
        ticker: Symbole boursier (ex: 'AAPL')
        prices: Instance PriceSeries contenant l'historique (COMPOSÉE)
        sector: Classification sectorielle optionnelle
        currency: Devise des prix (défaut: USD)
    
    Example:
        >>> ts = PriceSeries([100.0, 105.0, 110.0], "AAPL_prices")
        >>> apple = Asset("AAPL", ts, sector="Technology")
        >>> apple.volatility
        0.1587...
    """
    
    def __init__(
        self, 
        ticker: str, 
        prices: PriceSeries,
        sector: str | None = None,
        currency: CurrencyEnum = CurrencyEnum.USD
    ) -> None:
        """
        Initialise un Asset.
        
        Args:
            ticker: Symbole boursier (ne peut pas être vide)
            prices: Série de prix (ne peut pas être vide)
            sector: Secteur d'activité (optionnel)
            currency: Devise (défaut: USD)
        
        Raises:
            ValueError: Si ticker est vide ou prices est vide
        """
        # Validation des entrées dans le constructeur
        if not ticker or not ticker.strip():
            raise ValueError("Le ticker ne peut pas être vide")
        if len(prices) == 0:
            raise ValueError("La série de prix ne peut pas être vide")
        
        self.ticker = ticker.upper()  # Normalisation en majuscules
        self.prices = prices  # Composition : Asset POSSÈDE une PriceSeries
        self.sector = sector
        self.currency = currency
    
    def __repr__(self) -> str:
        """Représentation pour le développement."""
        return f"Asset({self.ticker!r}, {len(self.prices)} prices)"
    
    def __str__(self) -> str:
        """Représentation pour l'utilisateur."""
        return f"{self.ticker}: ${self.current_price:.2f}"
```

**Test dans le REPL :**

```python
>>> from pyvest.core.priceseries import PriceSeries
>>> from pyvest.core.asset import Asset
>>> ts = PriceSeries([100.0, 105.0, 103.0, 110.0], "AAPL_prices")
>>> apple = Asset("AAPL", ts, sector="Technology")
>>> apple
Asset('AAPL', 4 prices)
>>> apple.ticker
'AAPL'
>>> apple.prices.total_return
0.1
```

---

### Étape 2.2 : Propriétés et délégation vers PriceSeries

#### Le décorateur @property

Une **propriété** est un hybride entre un attribut et une méthode. Elle permet d'accéder à une valeur calculée avec la syntaxe d'un attribut (sans parenthèses), tout en exécutant du code derrière.

**Avantages des propriétés :**
- Syntaxe propre : `asset.volatility` au lieu de `asset.get_volatility()`
- Lecture seule par défaut (protection des données)
- Calcul à la demande (lazy evaluation)
- Possibilité d'ajouter de la validation via un setter

```python
# Ajout des propriétés à la classe Asset

class Asset:
    # ... (code précédent) ...
    
    @property
    def current_price(self) -> float:
        """Dernier prix connu."""
        return self.prices.values[-1]
    
    @property
    def volatility(self) -> float:
        """Volatilité annualisée (délègue à PriceSeries)."""
        return self.prices.annualized_volatility()
    
    @property
    def total_return(self) -> float:
        """Rendement total (délègue à PriceSeries)."""
        return self.prices.total_return
    
    @property
    def sharpe_ratio(self) -> float:
        """Ratio de Sharpe (délègue à PriceSeries)."""
        return self.prices.sharpe_ratio()
    
    @property
    def max_drawdown(self) -> float:
        """Drawdown maximum (délègue à PriceSeries)."""
        return self.prices.max_drawdown()
```

**Test de la délégation :**

```python
>>> apple = Asset("AAPL", PriceSeries([100.0, 105.0, 110.0], "AAPL"))
>>> apple.volatility        # Délègue à self.prices.annualized_volatility()
>>> apple.total_return      # Délègue à self.prices.total_return
0.1
```

---

#### Comprendre les décorateurs

Un **décorateur** est une fonction qui prend une autre fonction en argument et retourne une version modifiée de celle-ci. La syntaxe `@decorator` est un sucre syntaxique :

```python
@decorator
def ma_fonction():
    print("Ceci est ma fonction")

# Est équivalent à :
def ma_fonction():
    print("Ceci est ma fonction")
ma_fonction = decorator(ma_fonction)
```

**Points clés :**

1. **Exécution à l'import** : Le décorateur s'exécute quand le module est chargé, pas quand la fonction est appelée
2. **Closure** : La fonction interne du décorateur a accès aux variables de la fonction externe (scope `nonlocal`)
3. **Préservation des métadonnées** : Utiliser `@functools.wraps` pour conserver `__name__` et `__doc__`

**Exemple : décorateur de chronométrage**

```python
import time
import functools

def chronometre(func):
    """Décorateur qui mesure le temps d'exécution d'une fonction."""
    @functools.wraps(func)  # Préserve __name__ et __doc__
    def wrapper(*args, **kwargs):
        debut = time.perf_counter()
        resultat = func(*args, **kwargs)
        duree = time.perf_counter() - debut
        print(f"[{duree:.4f}s] {func.__name__}")
        return resultat
    return wrapper

@chronometre
def calcul_lent():
    time.sleep(1)
    return 42

>>> calcul_lent()
[1.0012s] calcul_lent
42
```

---

### Étape 2.3 : Méthode de corrélation entre actifs

#### Contexte en finance quantitative

La corrélation est une mesure fondamentale en finance quantitative. D'après *Portfolio Optimization: Theory and Application* (Palomar, 2025) :

> "Il existe une structure le long de la dimension des actifs (également appelée structure transversale). Cela signifie que plutôt que de considérer les actifs un par un de manière indépendante, ils doivent être modélisés conjointement. C'est particulièrement important pour évaluer le risque d'un portefeuille, car différentes actions peuvent avoir différentes corrélations."

Et d'après *The Elements of Quantitative Investing* (Paleologo) :

> "Les rendements ont généralement une corrélation positive car les actifs tendent à évoluer ensemble avec le marché."

La corrélation de Pearson mesure la relation linéaire entre deux variables. Pour deux actifs, nous calculons la corrélation de leurs log-rendements.

**Formule de Pearson :**

$$\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

**Interprétation :**
- **ρ = +1** : Corrélation positive parfaite (les actifs bougent ensemble)
- **ρ = 0** : Pas de corrélation linéaire
- **ρ = -1** : Corrélation négative parfaite (les actifs bougent en sens opposé)

```python
import numpy as np

class Asset:
    # ... (code précédent) ...
    
    def correlation_with(self, other: "Asset") -> float:
        """
        Calcule la corrélation de Pearson des log-rendements avec un autre actif.
        
        La corrélation mesure la relation linéaire entre deux séries.
        En finance, elle est cruciale pour :
        - Évaluer les bénéfices de diversification
        - Construire des portefeuilles optimaux
        - Identifier les paires de trading
        
        Args:
            other: Un autre Asset avec lequel calculer la corrélation
        
        Returns:
            Coefficient de corrélation entre -1 et 1
        
        Raises:
            ValueError: Si les séries ont moins de 2 observations communes
                        ou si une série a une variance nulle
        
        Note:
            Les séries sont alignées sur la longueur minimale.
            En production, on alignerait plutôt sur les dates.
        """
        # Récupération des log-rendements
        x = np.array(self.prices.all_log_returns())
        y = np.array(other.prices.all_log_returns())
        
        # Alignement des longueurs (gestion des séries de tailles différentes)
        n = min(len(x), len(y))
        if n < 2:
            raise ValueError(
                f"Pas assez d'observations communes: {n}. "
                "Minimum requis: 2."
            )
        
        x = x[:n]
        y = y[:n]
        
        # Centrage des valeurs (soustraction de la moyenne)
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        
        # Covariance (numérateur)
        covariance = np.dot(x_centered, y_centered)
        
        # Variances pour le dénominateur
        var_x = np.dot(x_centered, x_centered)
        var_y = np.dot(y_centered, y_centered)
        
        # Vérification de la variance nulle
        if var_x == 0 or var_y == 0:
            raise ValueError(
                "Variance nulle détectée. "
                "La corrélation n'est pas définie pour une série constante."
            )
        
        # Formule de Pearson : cov(x,y) / (std(x) * std(y))
        return covariance / np.sqrt(var_x * var_y)
```

> **⚠️ Note de correction** : La version originale retournait `0.0` silencieusement en cas de variance nulle ou de données insuffisantes. C'est problématique car une corrélation de 0 a une signification réelle (absence de relation linéaire). La version corrigée lève une `ValueError` explicite pour éviter les erreurs silencieuses.

---

### Étape 2.4 : Git Commit

```bash
git add pyvest/core/asset.py
git commit -m "feat(core): add Asset class with composition over PriceSeries"
```

---

### Étape 3 : DataLoader avec système de cache intelligent

#### Installation de yfinance

```bash
# Dans le terminal :
uv add yfinance

# Test dans le REPL :
>>> import yfinance as yf
>>> msft = yf.Ticker("MSFT")
>>> msft.info['shortName']
'Microsoft Corporation'
```

`yfinance` est une librairie open-source non officielle qui simplifie l'accès à l'API Yahoo Finance. Elle supporte les prix historiques, les données fondamentales et les données d'options.

> **⚠️ Avertissement légal** : yfinance ≠ permission d'utiliser les données Yahoo comme bon vous semble. Pour tout usage au-delà de la recherche personnelle légère, obtenez une licence appropriée.

---

#### Conception de l'interface du DataLoader

```
Interface publique :
├── fetch_single_ticker(ticker, price_col, dates) -> PriceSeries
├── fetch_multiple_tickers(tickers, price_col, dates) -> dict[str, PriceSeries]
└── clear_cache() -> int

Interface privée :
├── _get_cache_path(ticker, price_col, dates) -> Path
├── _check_date_overlap(...) -> tuple[str, Timestamp, Timestamp]
├── _load_from_cache(...) -> tuple[DataFrame, str, tuple]
└── _save_to_cache(...) -> None
```

**Pourquoi un système de cache intelligent ?**

| Raison | Explication |
|--------|-------------|
| **Performance** | Éviter les appels API redondants, chargement local rapide |
| **Rate limiting** | Les APIs ont souvent des limites de requêtes |
| **Reproductibilité** | Garantir l'utilisation du même dataset en recherche |
| **Mode hors-ligne** | Travailler sans connexion internet |
| **Économie de bande passante** | Ne télécharger que les données manquantes |

#### Architecture du cache avec gestion des chevauchements

Le système de cache gère quatre scénarios de correspondance temporelle :

```
Cas 1: EXACT - La requête correspond exactement au cache
Cache:    |-------------|
Requête:  |-------------|
→ Retourner directement les données en cache

Cas 2: CONTAINS - Le cache contient la période demandée
Cache:    |-----------------|
Requête:      |---------|
→ Découper (slice) les données en cache

Cas 3: OVERLAP_AFTER - Chevauchement à droite
Cache:    |---------|
Requête:      |-----------|
→ Fusionner cache + nouvelles données à droite

Cas 4: OVERLAP_BEFORE - Chevauchement à gauche
Cache:          |---------|
Requête:   |-----------|
→ Fusionner nouvelles données à gauche + cache

Cas 5: MISS - Aucun chevauchement
Cache:    |-----|
Requête:              |-----|
→ Télécharger toutes les données
```

---

#### Implémentation du DataLoader

```python
# Fichier: pyvest/data/loader.py

from pathlib import Path
from datetime import datetime
import logging
import pickle
from typing import Sequence

import yfinance as yf
import pandas as pd

from pyvest.core.priceseries import PriceSeries


class DataLoader:
    """
    Charge des données de marché depuis Yahoo Finance avec mise en cache intelligente.
    
    Le système de cache gère cinq scénarios de correspondance temporelle :
    1. EXACT : La requête correspond exactement aux données en cache
    2. CONTAINS : La requête est un sous-ensemble du cache
    3. OVERLAP_AFTER : Intersection partielle, fetch complémentaire à droite
    4. OVERLAP_BEFORE : Intersection partielle, fetch complémentaire à gauche
    5. MISS : Aucune donnée en cache, fetch complet nécessaire
    
    Attributes:
        cache_dir: Répertoire de stockage du cache
        logger: Logger pour le suivi des opérations
    
    Example:
        >>> loader = DataLoader(cache_dir=".cache")
        >>> ts = loader.fetch_single_ticker("AAPL", "Close", ("2024-01-01", "2024-06-01"))
        >>> len(ts)
        125
    """
    
    def __init__(self, cache_dir: str = ".cache") -> None:
        """
        Initialise le DataLoader.
        
        Args:
            cache_dir: Chemin du répertoire de cache (créé si inexistant)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_cache_path(
        self, 
        ticker: str, 
        price_col: str, 
        dates: tuple[str, str]
    ) -> Path:
        """
        Génère le chemin du fichier cache pour une requête donnée.
        
        Format: {ticker}_{price_col}_{start}_{end}.pkl
        """
        return self.cache_dir / f"{ticker}_{price_col}_{dates[0]}_{dates[1]}.pkl"

    def _check_date_overlap(
        self,
        cached_start: pd.Timestamp,
        cached_end: pd.Timestamp,
        req_start: pd.Timestamp,
        req_end: pd.Timestamp
    ) -> tuple[str, pd.Timestamp | None, pd.Timestamp | None]:
        """
        Détermine le type de chevauchement entre le cache et la requête.
        
        Args:
            cached_start: Date de début des données en cache
            cached_end: Date de fin des données en cache
            req_start: Date de début de la requête
            req_end: Date de fin de la requête
        
        Returns:
            tuple: (status, gap_start, gap_end)
            - status: "exact" | "contains" | "overlap_before" | "overlap_after" | "miss"
            - gap_start: Début de la période manquante (si overlap)
            - gap_end: Fin de la période manquante (si overlap)
        """
        # Cas MISS: Aucune intersection
        if cached_end < req_start or cached_start > req_end:
            return ("miss", None, None)

        # Cas EXACT: Correspondance parfaite
        if cached_start == req_start and cached_end == req_end:
            return ("exact", None, None)

        # Cas CONTAINS: Le cache englobe la requête
        if cached_start <= req_start and cached_end >= req_end:
            return ("contains", None, None)

        # Cas OVERLAP_AFTER: Le cache commence avant/au début mais finit avant la fin
        if cached_start <= req_start and cached_end < req_end:
            gap_start = cached_end + pd.Timedelta(days=1)
            gap_end = req_end
            return ("overlap_after", gap_start, gap_end)

        # Cas OVERLAP_BEFORE: Le cache commence après le début mais finit après/à la fin
        if cached_start > req_start and cached_end >= req_end:
            gap_start = req_start
            gap_end = cached_start - pd.Timedelta(days=1)
            return ("overlap_before", gap_start, gap_end)

        # Cas non couvert (le cache est un sous-ensemble strict de la requête)
        # Traité comme MISS pour simplicité - on refetch tout
        return ("miss", None, None)

    def _load_from_cache(
        self,
        ticker: str,
        price_col: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> tuple[pd.DataFrame | None, str, tuple | None]:
        """
        Recherche et charge les données disponibles en cache.
        
        Parcourt les fichiers du répertoire cache pour trouver une correspondance
        avec le couple (ticker, price_col) et détermine le type de chevauchement.
        
        Args:
            ticker: Symbole boursier
            price_col: Nom de la colonne prix ('Close', 'Open', etc.)
            start_date: Date de début de la requête
            end_date: Date de fin de la requête
        
        Returns:
            tuple: (dataframe, status, gap_range)
            - dataframe: Données en cache ou None
            - status: Type de correspondance
            - gap_range: (gap_start, gap_end) si overlap, sinon None
        """
        if not self.cache_dir.exists():
            return (None, "miss", None)

        # Itération sur les fichiers du cache pour match (ticker, price_col)
        for file_path in self.cache_dir.iterdir():
            if not file_path.is_file() or file_path.suffix != '.pkl':
                continue

            try:
                # Parse le nom du fichier: ticker_pricecol_startdate_enddate.pkl
                name_parts = file_path.stem.split('_')
                
                # Vérification du format attendu (4 parties minimum)
                if len(name_parts) < 4:
                    continue
                
                cached_ticker = name_parts[0]
                cached_col = name_parts[1]
                cached_start_str = name_parts[2]
                cached_end_str = name_parts[3]

                # Vérifier la correspondance ticker et price_col
                if cached_ticker != ticker or cached_col != price_col:
                    continue

                # Parser les dates
                cached_start = pd.to_datetime(cached_start_str)
                cached_end = pd.to_datetime(cached_end_str)

                # Déterminer le type de chevauchement
                status, gap_start, gap_end = self._check_date_overlap(
                    cached_start, cached_end, start_date, end_date
                )

                # Charger les données si le statut n'est pas 'miss'
                if status != "miss":
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)

                    # Reconstruire le DataFrame avec les dates
                    prices_list = data['prices']
                    dates_list = data.get('dates')  # Nouveau: stocker les dates réelles
                    
                    df = pd.DataFrame({price_col: prices_list})
                    
                    if dates_list is not None:
                        # Utiliser les dates réelles stockées
                        df.index = pd.to_datetime(dates_list)
                    else:
                        # Fallback: utiliser les jours ouvrés (moins précis)
                        date_range = pd.bdate_range(
                            start=cached_start, 
                            periods=len(df)
                        )
                        df.index = date_range

                    if status == "exact":
                        return (df, "exact", None)
                    elif status == "contains":
                        return (df, "contains", None)
                    elif status.startswith("overlap"):
                        return (df, status, (gap_start, gap_end))

            except (ValueError, KeyError, pickle.UnpicklingError) as e:
                # Ignorer les fichiers cache corrompus
                self.logger.warning(f"Fichier cache corrompu {file_path}: {e}")
                continue

        return (None, "miss", None)
    
    def _save_to_cache(
        self, 
        cache_path: Path, 
        prices: list[float],
        dates: list,
        ticker: str, 
        start: str, 
        end: str
    ) -> None:
        """
        Sauvegarde les prix dans un fichier cache avec métadonnées.
        
        Args:
            cache_path: Chemin du fichier cache
            prices: Liste des prix
            dates: Liste des dates correspondantes (index du DataFrame)
            ticker: Symbole boursier
            start: Date de début (string)
            end: Date de fin (string)
        """
        data = {
            "ticker": ticker,
            "start": start,
            "end": end,
            "fetched_at": datetime.now().isoformat(),
            "n_prices": len(prices),
            "prices": prices,
            "dates": dates  # Stocker les dates réelles pour reconstruction précise
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        self.logger.debug(f"Cache sauvegardé: {cache_path}")
    
    def fetch_single_ticker(
        self, 
        ticker: str, 
        price_col: str, 
        dates: tuple[str, str]
    ) -> PriceSeries | None:
        """
        Récupère les données de prix d'un ticker unique avec système de cache.
        
        Gère automatiquement les 5 cas de correspondance temporelle avec le cache.
        
        Args:
            ticker: Symbole boursier (ex: 'AAPL')
            price_col: Nom de la colonne prix (ex: 'Close', 'Open')
            dates: Tuple (start_date, end_date) au format 'YYYY-MM-DD'
        
        Returns:
            Instance de PriceSeries ou None si échec
        
        Example:
            >>> loader = DataLoader()
            >>> ts = loader.fetch_single_ticker("AAPL", "Close", ("2024-01-01", "2024-06-01"))
        """
        # Conversion des dates en Timestamp
        start_date = pd.to_datetime(dates[0], format="%Y-%m-%d")
        end_date = pd.to_datetime(dates[1], format="%Y-%m-%d")

        self.logger.info(f"Fetching {ticker} {price_col} from {dates[0]} to {dates[1]}")

        # Vérifier le cache
        cached_df, status, gap_range = self._load_from_cache(
            ticker, price_col, start_date, end_date
        )

        if status == "exact":
            self.logger.info(f"Cache HIT (exact): {ticker}")
            prices = cached_df[price_col].tolist()
            return PriceSeries(values=prices, name=price_col)

        elif status == "contains":
            self.logger.info(f"Cache HIT (slicing): {ticker}")
            # Découper le DataFrame cache pour obtenir le sous-ensemble demandé
            mask = (cached_df.index >= start_date) & (cached_df.index <= end_date)
            sliced_df = cached_df[mask]
            prices = sliced_df[price_col].tolist()
            return PriceSeries(values=prices, name=price_col)

        elif status.startswith("overlap"):
            self.logger.info(f"Cache PARTIAL (extending): {ticker}")
            gap_start, gap_end = gap_range

            # Fetch la partie manquante
            ticker_instance = yf.Ticker(ticker)
            gap_df = ticker_instance.history(start=gap_start, end=gap_end)

            if gap_df.empty:
                self.logger.warning(f"Échec du fetch des données manquantes pour {ticker}")
                return None

            # Fusionner le cache avec les nouvelles données
            if status == "overlap_after":
                # Concaténation à droite
                merged_df = pd.concat([cached_df[[price_col]], gap_df[[price_col]]])
            else:
                # Concaténation à gauche
                merged_df = pd.concat([gap_df[[price_col]], cached_df[[price_col]]])

            # Supprimer les doublons et trier par date
            merged_df = merged_df[~merged_df.index.duplicated(keep='first')].sort_index()

            # Sauvegarder le cache étendu
            cache_path = self._get_cache_path(ticker, price_col, dates)
            prices_list = merged_df[price_col].tolist()
            dates_list = merged_df.index.tolist()
            
            self._save_to_cache(
                cache_path=cache_path,
                prices=prices_list,
                dates=dates_list,
                ticker=ticker,
                start=dates[0],
                end=dates[1]
            )

            return PriceSeries(values=prices_list, name=price_col)

        else:  # miss: pas de données en cache correspondantes
            self.logger.info(f"Cache MISS: fetching from Yahoo Finance for {ticker}")

            ticker_instance = yf.Ticker(ticker)
            df = ticker_instance.history(start=start_date, end=end_date)

            if df.empty:
                self.logger.warning(f"DataFrame vide retourné pour {ticker}")
                return None

            if price_col not in df.columns:
                self.logger.error(f"Colonne '{price_col}' non trouvée dans les données")
                raise KeyError(f"Colonne '{price_col}' non disponible")

            prices = df[price_col].tolist()
            dates_list = df.index.tolist()

            if not prices:
                self.logger.warning(f"Liste de prix vide pour {ticker}[{price_col}]")
                return None

            # Sauvegarder dans le cache
            cache_path = self._get_cache_path(ticker, price_col, dates)
            self._save_to_cache(
                cache_path=cache_path,
                prices=prices,
                dates=dates_list,
                ticker=ticker,
                start=dates[0],
                end=dates[1]
            )

            return PriceSeries(values=prices, name=price_col)
    
    def fetch_multiple_tickers(
        self,
        tickers: Sequence[str],
        price_col: str,
        dates: tuple[str, str]
    ) -> dict[str, PriceSeries]:
        """
        Récupère les données de prix pour plusieurs tickers.
        
        Args:
            tickers: Liste de symboles boursiers
            price_col: Nom de la colonne prix
            dates: Tuple (start_date, end_date)
        
        Returns:
            Dictionnaire {ticker: PriceSeries}
            Les tickers en échec sont omis du résultat.
        """
        results = {}
        for ticker in tickers:
            ps = self.fetch_single_ticker(ticker, price_col, dates)
            if ps is not None:
                results[ticker] = ps
        return results
    
    def clear_cache(self) -> int:
        """
        Supprime tous les fichiers du cache.
        
        Returns:
            Nombre de fichiers supprimés
        """
        count = 0
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
            count += 1
        self.logger.info(f"Cache vidé: {count} fichiers supprimés")
        return count
```

> **⚠️ Corrections apportées au système de cache :**
>
> 1. **Stockage des dates réelles** : La version originale reconstruisait les dates avec `freq='B'` (jours ouvrés), ce qui ne correspond pas aux jours de trading réels (ne tient pas compte des jours fériés). La version corrigée stocke les dates réelles du DataFrame yfinance.
>
> 2. **Gestion robuste du parsing** : Ajout d'une vérification du nombre de parties dans le nom de fichier.
>
> 3. **Cas manquant identifié** : Le cas où le cache est un sous-ensemble strict de la requête (cache inclus dans requête) est traité comme MISS pour éviter une logique complexe de double fetch.

---

#### Test du système de cache

```python
from pyvest.data.loader import DataLoader
import time

loader = DataLoader(cache_dir=".cache")

# Premier appel : fetch depuis l'API
start = time.perf_counter()
ts1 = loader.fetch_single_ticker("AAPL", "Close", ("2024-01-01", "2024-06-01"))
premier_temps = time.perf_counter() - start
print(f"Premier fetch: {premier_temps:.2f} secondes")

# Second appel : chargement depuis le cache
start = time.perf_counter()
ts2 = loader.fetch_single_ticker("AAPL", "Close", ("2024-01-01", "2024-06-01"))
second_temps = time.perf_counter() - start
print(f"Second fetch: {second_temps:.4f} secondes")

print(f"Accélération: {premier_temps/second_temps:.0f}x plus rapide avec le cache")

# Test d'extension du cache
start = time.perf_counter()
ts3 = loader.fetch_single_ticker("AAPL", "Close", ("2024-01-01", "2024-09-01"))
extension_temps = time.perf_counter() - start
print(f"Extension du cache: {extension_temps:.2f} secondes")
```

---

### Étape 4 : Collections Python

#### Les listes : collections ordonnées

D'après *Fluent Python* (Ramalho) :

> "Une liste en Python est une séquence mutable d'objets arbitraires."

Les listes sont la collection de base en Python. Elles maintiennent l'ordre d'insertion, supportent l'indexation et le slicing, et peuvent contenir n'importe quel type d'objet.

**Caractéristiques principales :**

| Propriété | Description |
|-----------|-------------|
| **Mutable** | On peut ajouter, supprimer ou modifier des éléments |
| **Ordonnée** | L'ordre d'insertion est préservé |
| **Indexable** | Accès par position `[i]` et slicing `[a:b]` |
| **Hétérogène** | Peut contenir des types différents |

```python
# Opérations courantes sur les listes
assets = [apple, msft, google, amazon]

# Accès par index
premier = assets[0]           # Premier élément
dernier = assets[-1]          # Dernier élément
sous_liste = assets[1:3]      # Éléments aux indices 1 et 2

# Modification
assets.append(nvidia)         # Ajout en fin
assets.insert(0, tesla)       # Insertion à une position
assets.remove(msft)           # Suppression par valeur
element = assets.pop()        # Suppression et retour du dernier

# Itération
for asset in assets:
    print(f"{asset.ticker}: {asset.volatility:.2%}")

# Recherche
if apple in assets:
    print("Apple est dans la liste")
```

---

#### Syntaxe des compréhensions

D'après *Fluent Python* :

> "Une compréhension de liste est plus explicite. Son but est toujours de construire une nouvelle liste."

Les compréhensions offrent une syntaxe concise pour créer des collections à partir d'itérables existants.

**Anatomie d'une compréhension :**

```python
[expression for element in iterable if condition]
 ↑           ↑                       ↑
 sortie      boucle                  filtre optionnel
```

**Exemples pratiques :**

```python
# Extraction des tickers
tickers = [asset.ticker for asset in assets]

# Tuples (ticker, rendement)
performance = [(a.ticker, a.total_return) for a in assets]

# Filtrage : actifs avec rendement positif
gagnants = [a for a in assets if a.total_return > 0]

# Filtrage : haute volatilité (> 25%)
haute_vol = [a.ticker for a in assets if a.volatility > 0.25]

# Tri par rendement (décroissant)
tries = sorted(assets, key=lambda a: a.total_return, reverse=True)
```

**Expressions génératrices** : Utilisent des parenthèses au lieu de crochets et produisent les éléments paresseusement (un à la fois), économisant la mémoire :

```python
# Générateur : ne crée pas de liste en mémoire
total_vol = sum(a.volatility for a in assets)

# Comparaison de l'utilisation mémoire :
sum([a.volatility for a in assets])  # Crée une liste intermédiaire
sum(a.volatility for a in assets)    # Pas de liste intermédiaire (lazy)
```

---

#### Les dictionnaires : mapping clé-valeur

D'après *Fluent Python* :

> "Le type dict est une brique fondamentale de Python."

Les dictionnaires offrent une recherche en O(1) par clé, ce qui les rend essentiels pour un accès rapide aux données.

**Caractéristiques principales :**

| Propriété | Description |
|-----------|-------------|
| **Clés uniques** | Chaque clé apparaît une seule fois |
| **Clés hashables** | Les clés doivent être immutables (str, int, tuple) |
| **Recherche O(1)** | Accès quasi-instantané par clé |
| **Ordre préservé** | Depuis Python 3.7, l'ordre d'insertion est garanti |

```python
# Création par compréhension : table de correspondance ticker → asset
assets_par_ticker = {asset.ticker: asset for asset in assets}

# Recherche rapide O(1)
apple = assets_par_ticker["AAPL"]
nvidia = assets_par_ticker.get("NVDA")  # Sûr : retourne None si absent

# Vérification d'existence
if "TSLA" in assets_par_ticker:
    tesla = assets_par_ticker["TSLA"]

# Itération sur les paires clé-valeur
for ticker, asset in assets_par_ticker.items():
    print(f"{ticker}: ${asset.current_price:.2f}")
```

---

#### Les ensembles : collections uniques

D'après *Fluent Python* :

> "Un ensemble est une collection d'objets uniques."

Les ensembles excellent pour le test d'appartenance (O(1) vs O(n) pour les listes) et les opérations ensemblistes.

```python
# Comparaison de portefeuilles avec opérations ensemblistes
mon_portefeuille = {"AAPL", "MSFT", "GOOGL"}
benchmark = {"AAPL", "MSFT", "AMZN", "NVDA", "META"}

# Intersection : présent dans les deux
commun = mon_portefeuille & benchmark  # {'AAPL', 'MSFT'}

# Différence : dans le benchmark mais pas dans mon portefeuille
manquant = benchmark - mon_portefeuille  # {'AMZN', 'NVDA', 'META'}

# Test d'appartenance rapide O(1)
"AAPL" in mon_portefeuille  # True (instantané)
```

---

## Partie 2 : Exercices Pratiques

---

### Exercice 1 : La classe Universe (25 min)

**Objectif :** Créer une classe `Universe` pour gérer une collection d'actifs avec le pattern d'**agrégation**.

```python
# Fichier: pyvest/core/universe.py

from pyvest.core.asset import Asset
from typing import Iterator


class Universe:
    """
    Collection d'actifs représentant un univers d'investissement.
    
    Pattern de conception : AGRÉGATION
    ──────────────────────────────────
    Universe CONTIENT des Asset, mais les Asset peuvent exister 
    indépendamment de l'Universe.
    
    La classe implémente le protocole d'itération (__iter__) et
    de conteneur (__contains__, __len__) pour une utilisation
    pythonique.
    """
    
    def __init__(self, assets: list[Asset] | None = None) -> None:
        self._assets: dict[str, Asset] = {}
        if assets:
            for asset in assets:
                self.add(asset)
    
    def add(self, asset: Asset) -> None:
        """Ajoute un actif à l'univers."""
        # Votre code ici
        pass
    
    def get(self, ticker: str) -> Asset | None:
        """Récupère un actif par son ticker."""
        # Votre code ici
        pass
    
    def remove(self, ticker: str) -> Asset | None:
        """Retire un actif de l'univers."""
        # Votre code ici
        pass
    
    def __len__(self) -> int:
        # Votre code ici
        pass
    
    def __iter__(self) -> Iterator[Asset]:
        # Votre code ici
        pass
    
    def __contains__(self, ticker: str) -> bool:
        # Votre code ici
        pass
    
    @property
    def tickers(self) -> list[str]:
        # Votre code ici
        pass
    
    def filter_by_sector(self, sector: str) -> list[Asset]:
        """Filtre les actifs par secteur. Utilisez une compréhension !"""
        # Votre code ici
        pass
```

---

### Exercice 2 : Top K Corrélations (30 min)

**Objectif :** Implémenter une fonction qui extrait les K paires d'actifs les plus corrélées d'un univers.

#### Contexte en finance quantitative

L'extraction des paires les plus corrélées est une analyse fondamentale en finance quantitative. D'après *Portfolio Optimization* (Palomar, 2025) :

> "La diversification d'un investissement en allouant du capital à plusieurs actifs peut ne pas aider à réduire le risque si ces actifs sont fortement corrélés."

**Applications pratiques de l'analyse des top-K corrélations :**

| Application | Description |
|-------------|-------------|
| **Détection de risque de concentration** | Identifier les paires très corrélées qui augmentent le risque systémique |
| **Pairs trading** | Trouver des paires d'actifs co-intégrés pour des stratégies de mean-reversion |
| **Construction de portefeuille** | Éviter la sur-représentation de facteurs cachés |
| **Monitoring de régime** | Suivre l'évolution des corrélations dans le temps |
| **Stress testing** | Identifier les liens qui pourraient amplifier les pertes en période de crise |

> **Note pédagogique** : Cette fonction sera remplacée par une version vectorisée plus performante dans la Session 3 (`Universe.top_correlations()`), mais l'implémentation avec boucles permet de bien comprendre l'algorithme sous-jacent.

**Référence méthodologique :** Cette analyse est inspirée du projet "Global Multi-Asset Correlation Lab" qui utilise une approche similaire pour analyser les corrélations entre actifs de différentes classes.

```python
from itertools import combinations


def top_k_correlations(
    assets: list[Asset],
    k: int = 20,
    use_absolute: bool = False
) -> list[tuple[str, str, float]]:
    """
    Extrait les K paires les plus corrélées d'une liste d'actifs.
    
    Cette fonction calcule la corrélation de Pearson entre toutes les
    paires possibles d'actifs et retourne les K paires avec les plus
    fortes corrélations.
    
    Args:
        assets: Liste d'objets Asset
        k: Nombre de paires à retourner (défaut: 20)
        use_absolute: Si True, trie par |corrélation| pour capturer
                      aussi les fortes corrélations négatives
    
    Returns:
        Liste de tuples (ticker_1, ticker_2, corrélation) triée
        par corrélation décroissante
    
    Complexity:
        O(n²) où n = nombre d'actifs (toutes les paires sont calculées)
    
    Example:
        >>> pairs = top_k_correlations(assets, k=5)
        >>> pairs[0]
        ('AAPL', 'MSFT', 0.85)
    
    Note:
        Cette implémentation sera optimisée en Session 3 avec
        des opérations matricielles vectorisées.
    """
    correlations = []
    
    # itertools.combinations génère toutes les paires uniques
    # évitant les doublons (A,B) et (B,A) et les auto-corrélations (A,A)
    for asset_1, asset_2 in combinations(assets, 2):
        # Calculer la corrélation
        # Votre code ici...
        pass
    
    # Trier par corrélation (ou valeur absolue) et retourner les k premières
    # Votre code ici...
    pass
```

---

### Exercice 3 : Construction d'une matrice de corrélation (30 min)

**Objectif :** Créer une fonction qui construit une matrice de corrélation complète.

#### Contexte théorique

La **matrice de corrélation** est une composante essentielle de l'analyse de portefeuille. D'après *The Elements of Quantitative Investing* (Paleologo) :

> "La variance du portefeuille (risque) dépend à la fois des volatilités individuelles ET des corrélations. La matrice de covariance capture la façon dont les actifs évoluent ensemble."

Et d'après *Portfolio Optimization* (Palomar, 2025) :

> "Les deux composantes principales de la conception de portefeuille sont la modélisation des données et l'optimisation du portefeuille. [...] Le but principal du bloc de modélisation est de caractériser la distribution statistique des rendements futurs, principalement en termes de moments de premier et second ordre (μ et Σ)."

**Propriétés de la matrice de corrélation :**
- **Symétrique** : corr(A,B) = corr(B,A)
- **Diagonale = 1** : corr(A,A) = 1 pour tout actif A
- **Valeurs dans [-1, 1]** : par définition de la corrélation de Pearson
- **Semi-définie positive** : toute combinaison linéaire a une variance ≥ 0

```python
import pandas as pd
import numpy as np
from itertools import combinations


def build_correlation_matrix(assets: list[Asset]) -> pd.DataFrame:
    """
    Construit une matrice de corrélation pour tous les actifs.
    
    La matrice est symétrique : corr(A,B) = corr(B,A)
    La diagonale vaut 1 : corr(A,A) = 1
    
    Args:
        assets: Liste d'objets Asset
    
    Returns:
        DataFrame symétrique avec tickers en index et colonnes
    
    Example:
        >>> matrix = build_correlation_matrix([apple, msft, google])
        >>> matrix.loc['AAPL', 'MSFT']
        0.85
    """
    tickers = [a.ticker for a in assets]
    n = len(tickers)
    
    # Initialiser la matrice avec NaN
    matrix = np.full((n, n), np.nan)
    
    # Remplir la diagonale avec 1.0 (auto-corrélation)
    np.fill_diagonal(matrix, 1.0)
    
    # Créer un mapping ticker -> index pour un accès rapide O(1)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    
    # Remplir le triangle supérieur et inférieur (symétrie)
    # Votre code ici...
    
    return pd.DataFrame(matrix, index=tickers, columns=tickers)


def extract_upper_triangle(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait les paires uniques du triangle supérieur de la matrice.
    
    Utile pour éviter les doublons (AAPL-MSFT et MSFT-AAPL) et
    exclure la diagonale (auto-corrélations).
    
    Cette méthode est similaire à celle utilisée dans le projet
    "Global Multi-Asset Correlation Lab".
    
    Args:
        corr_matrix: Matrice de corrélation (DataFrame carré)
    
    Returns:
        DataFrame avec colonnes ['asset_1', 'asset_2', 'correlation']
        trié par corrélation décroissante
    """
    # Créer un masque pour le triangle supérieur (excluant la diagonale k=1)
    mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    
    # Votre code ici...
    pass
```

---

### Exercice 4 : Statistiques de l'univers (25 min)

**Objectif :** Ajouter une méthode `summary()` à la classe Universe.

```python
def summary(self) -> pd.DataFrame:
    """
    Génère un résumé statistique de tous les actifs de l'univers.
    
    Returns:
        DataFrame avec une ligne par actif et les colonnes:
        - ticker
        - sector
        - current_price
        - total_return
        - volatility
        - sharpe_ratio
        - max_drawdown
    """
    # Votre code ici...
    pass
```

---

## Solutions

### Solution - Exercice 1

```python
class Universe:
    def __init__(self, assets: list[Asset] | None = None) -> None:
        self._assets: dict[str, Asset] = {}
        if assets:
            for asset in assets:
                self.add(asset)
    
    def add(self, asset: Asset) -> None:
        self._assets[asset.ticker] = asset
    
    def get(self, ticker: str) -> Asset | None:
        return self._assets.get(ticker)
    
    def remove(self, ticker: str) -> Asset | None:
        return self._assets.pop(ticker, None)
    
    def __len__(self) -> int:
        return len(self._assets)
    
    def __iter__(self) -> Iterator[Asset]:
        return iter(self._assets.values())
    
    def __contains__(self, ticker: str) -> bool:
        return ticker in self._assets
    
    def __repr__(self) -> str:
        return f"Universe({len(self)} assets: {self.tickers})"
    
    @property
    def tickers(self) -> list[str]:
        return list(self._assets.keys())
    
    def filter_by_sector(self, sector: str) -> list[Asset]:
        return [a for a in self._assets.values() if a.sector == sector]
```

---

### Solution - Exercice 2

```python
from itertools import combinations

def top_k_correlations(
    assets: list[Asset],
    k: int = 20,
    use_absolute: bool = False
) -> list[tuple[str, str, float]]:
    """
    Extrait les K paires les plus corrélées.
    
    Algorithme similaire au projet "Global Multi-Asset Correlation Lab"
    mais adapté pour travailler avec des objets Asset plutôt qu'un DataFrame.
    """
    correlations = []
    
    # Générer toutes les paires uniques avec combinations
    for asset_1, asset_2 in combinations(assets, 2):
        try:
            corr = asset_1.correlation_with(asset_2)
            correlations.append((asset_1.ticker, asset_2.ticker, corr))
        except ValueError:
            # Ignorer les paires avec données insuffisantes
            continue
    
    # Fonction de tri : valeur absolue ou valeur brute
    if use_absolute:
        key_func = lambda x: abs(x[2])
    else:
        key_func = lambda x: x[2]
    
    # Trier par corrélation décroissante et retourner les k premières
    return sorted(correlations, key=key_func, reverse=True)[:k]
```

> **Comparaison avec le Global Multi-Asset Correlation Lab :**
> 
> Le projet de référence utilise une approche matricielle :
> ```python
> # Version matricielle (Correlation Lab)
> def top_k_correlations(corr: pd.DataFrame, k: int=20, use_abs: bool=False):
>     m = corr.copy()
>     np.fill_diagonal(m.values, np.nan)
>     upper = m.where(np.triu(np.ones(m.shape), 1).astype(bool))
>     df = upper.stack().reset_index()
>     df.columns = ["col_1", "col_2", "correlation"]
>     key = df["correlation"].abs() if use_abs else df["correlation"]
>     return df.assign(_rk=key).sort_values("_rk", ascending=False).head(k)
> ```
> 
> Notre version avec boucles est pédagogiquement plus claire, mais sera optimisée en Session 3 avec une approche similaire à la version matricielle.

---

### Solution - Exercice 3

```python
def build_correlation_matrix(assets: list[Asset]) -> pd.DataFrame:
    """Construit une matrice de corrélation pour tous les actifs."""
    tickers = [a.ticker for a in assets]
    n = len(tickers)
    
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 1.0)
    
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    
    for a1, a2 in combinations(assets, 2):
        try:
            corr = a1.correlation_with(a2)
            i, j = ticker_to_idx[a1.ticker], ticker_to_idx[a2.ticker]
            matrix[i, j] = corr
            matrix[j, i] = corr  # Symétrie
        except ValueError:
            continue
    
    return pd.DataFrame(matrix, index=tickers, columns=tickers)


def extract_upper_triangle(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait les paires uniques du triangle supérieur.
    
    Méthode identique à celle du Global Multi-Asset Correlation Lab.
    """
    # Masque pour le triangle supérieur (k=1 exclut la diagonale)
    mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    
    # Appliquer le masque et convertir en format long
    upper = corr_matrix.where(mask)
    pairs = upper.stack().reset_index()
    pairs.columns = ['asset_1', 'asset_2', 'correlation']
    
    # Trier par corrélation décroissante
    return pairs.sort_values('correlation', ascending=False).reset_index(drop=True)
```

---

### Solution - Exercice 4

```python
def summary(self) -> pd.DataFrame:
    """Génère un résumé statistique de l'univers."""
    data = []
    
    for asset in self._assets.values():
        data.append({
            'ticker': asset.ticker,
            'sector': asset.sector,
            'current_price': asset.current_price,
            'total_return': asset.total_return,
            'volatility': asset.volatility,
            'sharpe_ratio': asset.sharpe_ratio,
            'max_drawdown': asset.max_drawdown
        })
    
    df = pd.DataFrame(data)
    
    # Trier par rendement total décroissant
    return df.sort_values('total_return', ascending=False).reset_index(drop=True)
```

---

## Résumé de la Session 2

### Concepts Python appris

| Concept | Application |
|---------|-------------|
| Composition | Asset POSSÈDE une PriceSeries |
| Agrégation | Universe CONTIENT des Asset |
| Décorateur `@property` | Attributs calculés en lecture seule |
| Décorateurs personnalisés | Chronométrage, logging |
| Listes | Collections ordonnées et mutables |
| Dictionnaires | Mapping clé-valeur O(1) |
| Ensembles | Collections uniques, opérations ensemblistes |
| Compréhensions | Création concise de collections |
| Expressions génératrices | Évaluation paresseuse |
| Système de cache | Persistance et optimisation des appels API |

### Concepts de finance quantitative

| Concept | Description | Référence |
|---------|-------------|-----------|
| Corrélation de Pearson | Mesure de relation linéaire entre actifs | *Portfolio Optimization* Ch. 2 |
| Matrice de corrélation | Vue d'ensemble des relations dans un univers | *Elements of QI* Ch. 8 |
| Top-K corrélations | Identification des paires les plus liées | Correlation Lab |
| Structure transversale | Modélisation conjointe des actifs | *Portfolio Optimization* §2.5 |

### Continuité pédagogique

Cette session pose les bases qui seront étendues dans les sessions suivantes :

| Session 2 (Boucles) | Session 3 (Vectorisé) |
|---------------------|----------------------|
| `top_k_correlations(assets)` | `universe.top_correlations()` |
| `build_correlation_matrix(assets)` | `universe.correlation_matrix()` |
| Boucles O(n²) | Opérations matricielles NumPy |

---

## Travail à faire

1. **Étendre Universe** : Ajouter une méthode `covariance_matrix()` qui retourne la matrice de covariance
2. **Expiration du cache** : Modifier DataLoader pour ignorer les fichiers cache de plus de N jours
3. **Lecture préparatoire** : Lire la documentation de NumPy et les chapitres sur les opérations matricielles

---

## Références

1. Lott, S. & Phillips, D. (2021). *Python Object-Oriented Programming*, 4th Edition. Packt.
2. Ramalho, L. (2022). *Fluent Python*, 2nd Edition. O'Reilly.
3. Palomar, D. P. (2025). *Portfolio Optimization: Theory and Application*. Cambridge University Press.
4. Paleologo, G. (2024). *The Elements of Quantitative Investing*. Chapman & Hall/CRC.
5. Documentation yfinance : https://ranaroussi.github.io/yfinance/
6. Diederich, J. (2012). "Stop Writing Classes" - PyCon 2012
7. Global Multi-Asset Correlation Lab - Complexity Explorer Project
