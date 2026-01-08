# Session 1 : Les Fondamentaux de Python pour le Développement

## Objectifs de la session

1. Créer une classe Python et comprendre les méthodes spéciales : `__init__`, `__repr__`, `__str__` et `__len__`
2. Comprendre la différence entre une classe et son instance
3. Utiliser les type hints et comprendre les types de données
4. Implémenter des fonctions et méthodes
5. Maîtriser les rudiments de Git

---

## Contexte du projet

Tout au long de ce cours, nous utiliserons un projet de création de librairie Python comme prétexte pour découvrir les concepts de développement pythoniques essentiels. Cette librairie sera basée sur des concepts classiques en gestion de portefeuille.

---

## Partie 1 : Développement Guidé

---

### Étape 1.0 : Configuration du projet avec Git

Avant de commencer à coder, configurons notre environnement de développement et initialisons notre dépôt Git.

```bash
# 1. Créer le projet avec uv
uv init pyvest --lib

# 2. Se placer dans le répertoire du projet
cd pyvest

# 3. Initialiser le dépôt Git local
git init --initial-branch main

# 4. Configurer votre identité Git (une seule fois par système)
git config --global user.name "VotreNom"
git config --global user.email "votre.email@example.com"

# 5. Vérifier la configuration
git config --global --list

# 6. Créer le premier commit
git add .
git commit -m "Initial commit: project structure"

# 7. (Optionnel) Lier à un dépôt distant après création sur GitHub
# git remote add origin https://github.com/username/pyvest.git
# git push -u origin main
```

---

### Étape 1.1 : L'objectif final de cette session

Voici la classe que nous allons construire progressivement au cours de cette session :

```python
import math

class PriceSeries:
    """
    Représentation d'une série temporelle de prix financiers.
    
    Attributes:
        values: Liste de prix indexés par le temps
        name: Identifiant de la série
    
    Class Attributes:
        TRADING_DAYS_PER_YEAR: Constante d'annualisation 
        (convention US equities, peut varier selon l'actif)
    """
    
    TRADING_DAYS_PER_YEAR: int = 252
    
    def __init__(self, values: list[float], name: str = "unnamed") -> None:
        self.values = list(values)  # Copie défensive
        self.name = name
    
    def __repr__(self) -> str:
        return f"PriceSeries({self.name!r}, {len(self.values)} values)"
    
    def __str__(self) -> str:
        if self.values:
            return f"{self.name}: {self.values[-1]:.2f} (latest)"
        return f"{self.name}: empty"
    
    def __len__(self) -> int:
        return len(self.values)
    
    def linear_return(self, t: int) -> float:
        """
        Calcule le rendement linéaire (arithmétique) entre t-1 et t.
        
        - Non ajusté des dividendes (utiliser le prix ajusté pour cela)
        - Additif entre actifs : r_portfolio = Σ(weight_i × r_i)
        
        Args:
            t: Position temporelle (doit être >= 1)

        Returns:
            Rendement en décimal (0.05 = 5%)
        """
        return (self.values[t] - self.values[t-1]) / self.values[t-1]
    
    def log_return(self, t: int) -> float:
        """
        Calcule le log-rendement entre t-1 et t.
        
        - Additif dans le temps : Σ(log returns) = log(P_T / P_0)
        - Permet d'approximer la variance multipériode par la somme des variances
        
        Args:
            t: Position temporelle (doit être >= 1)

        Returns:
            Log-rendement
        """
        return math.log(self.values[t] / self.values[t-1])
    
    @property
    def total_return(self) -> float:
        """
        Rendement total (non annualisé) sur toute la période.
        """
        if len(self.values) < 2:
            return 0.0
        return (self.values[-1] - self.values[0]) / self.values[0]
```

---

### Étape 1.2 : Structure du projet

La structure d'un tel projet pose forcément question. Utilisons le gestionnaire de projet `uv` pour initialiser notre librairie :

```bash
uv init pyvest --lib --package
```

Cette commande crée la structure suivante :

```
pyvest/
├── pyproject.toml      # Configuration du projet
├── README.md
└── src/
    └── pyvest/
        ├── __init__.py # Fait de pyvest un package
        └── ...         # Nos modules iront ici
```

Créons ensuite notre environnement virtuel :

```bash
uv venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

---

### Concepts fondamentaux : Module, Package et Librairie

> **Module** : Un fichier Python (`.py`) contenant du code (fonctions, classes, variables) qui peut être importé par d'autres fichiers.

> **Package** : Un répertoire contenant un fichier `__init__.py` et potentiellement d'autres modules ou sous-packages. Le `__init__.py` indique à Python que ce répertoire doit être traité comme un package importable.

> **Librairie** : Une collection de code créée par un tiers, composée d'un ou plusieurs packages. L'utilisateur final n'a pas besoin de connaître le code interne ; il utilise simplement l'interface publique (API).

> **API (Application Programming Interface)** : L'interface publique d'une application. Elle définit comment d'autres programmes peuvent communiquer avec votre code (appeler des fonctions, instancier des classes, etc.).

---

### Étape 1.3 : Création de la première classe

Créons notre premier fichier :

```bash
touch src/pyvest/priceseries.py
```

```python
# src/pyvest/priceseries.py

class PriceSeries:
    pass
```

Testons dans le REPL Python :

```python
>>> from pyvest.priceseries import PriceSeries
>>> ts = PriceSeries()
>>> ts
<pyvest.priceseries.PriceSeries object at 0x...>
>>> type(ts)
<class 'pyvest.priceseries.PriceSeries'>
```

---

### Concept fondamental : Qu'est-ce qu'une classe ?

Une **classe** est un patron (template) qui définit la structure et le comportement d'un type d'objet. On peut la concevoir comme une usine capable de fabriquer des objets selon un modèle défini.

Les objets créés à partir d'une classe sont appelés **instances**. Chaque instance possède :
- Des **attributs** : variables propres à l'instance
- Des **méthodes** : fonctions ayant accès aux attributs de l'instance

```python
# PriceSeries est la classe (le patron)
# ts1 et ts2 sont des instances (les objets créés)
ts1 = PriceSeries([100, 105, 110], "AAPL")
ts2 = PriceSeries([50, 52, 51], "MSFT")

# Chaque instance a ses propres données
ts1.name  # "AAPL"
ts2.name  # "MSFT"
```

> **Note sur `pass`** : C'est une instruction qui ne fait rien. Elle permet de définir des structures vides (classes, fonctions) que l'on complétera plus tard.

---

### Concept fondamental : Expression vs Statement (Instruction)

Cette distinction est fondamentale pour comprendre Python.

Une **expression** est une combinaison de valeurs, variables et opérateurs qui s'évalue pour produire une valeur :

```python
# Expressions - chacune produit une valeur
3 + 4              # → 7
x * 2              # → valeur numérique
len(my_list)       # → entier
a > b              # → True ou False
"hello".upper()    # → "HELLO"
```

Un **statement** (instruction) est une unité de code qui effectue une action mais ne produit pas de valeur utilisable :

```python
# Statements - effectuent des actions
if condition:      # Contrôle de flux
    pass
for x in items:    # Boucle
    pass
def ma_fonction(): # Définition de fonction
    pass
return valeur      # Retour de fonction
import math        # Import
x = 5              # Assignation
```

> **Cas particulier (Python 3.8+)** : L'opérateur morse `:=` (walrus operator) permet une assignation qui est aussi une expression :
> ```python
> # Assignation classique (statement)
> n = len(data)
> if n > 10:
>     print(n)
> 
> # Avec walrus operator (expression)
> if (n := len(data)) > 10:
>     print(n)
> ```

---

### Étape 1.4 : La méthode `__init__`

La méthode `__init__` est l'**initialiseur** de la classe. Elle est appelée automatiquement après la création de l'instance pour initialiser ses attributs.

> **Note technique** : On appelle souvent `__init__` le "constructeur" par abus de langage. En réalité, le véritable constructeur en Python est `__new__`, qui crée l'instance. `__init__` reçoit cette instance déjà créée et l'initialise. Pour la plupart des cas d'usage, cette distinction n'a pas d'importance pratique.

```python
class PriceSeries:
    
    TRADING_DAYS_PER_YEAR: int = 252  # Attribut de classe
    
    def __init__(self, values: list[float], name: str = "unnamed") -> None:
        self.values = list(values)  # Attribut d'instance (copie défensive)
        self.name = name            # Attribut d'instance
```

Le paramètre `self` est une référence à l'instance spécifique en cours de création. Python le passe automatiquement ; vous ne le fournissez pas lors de l'appel.

```python
# Ce que vous écrivez :
ts = PriceSeries([100, 105], "AAPL")

# Ce que Python exécute en coulisses (approximativement) :
# 1. Création de l'instance via __new__
instance = object.__new__(PriceSeries)
# 2. Initialisation via __init__
PriceSeries.__init__(instance, [100, 105], "AAPL")
# 3. Assignation à la variable
ts = instance
```

**Test dans le REPL :**

```python
>>> ts = PriceSeries([100.0, 102.5, 101.0, 105.0], "AAPL")
>>> ts.values
[100.0, 102.5, 101.0, 105.0]
>>> ts.name
'AAPL'
>>> ts.TRADING_DAYS_PER_YEAR  # Accessible via l'instance
252
>>> PriceSeries.TRADING_DAYS_PER_YEAR  # Ou via la classe
252

# Test avec valeur par défaut
>>> ts2 = PriceSeries([50.0, 51.0])
>>> ts2.name
'unnamed'
```

---

### Concept fondamental : Attributs de classe vs Attributs d'instance

| Type | Définition | Partage | Exemple |
|------|------------|---------|---------|
| **Attribut de classe** | Défini directement dans la classe | Partagé par toutes les instances | `TRADING_DAYS_PER_YEAR = 252` |
| **Attribut d'instance** | Défini via `self.xxx` dans `__init__` | Propre à chaque instance | `self.values = values` |

```python
>>> ts1 = PriceSeries([100], "A")
>>> ts2 = PriceSeries([200], "B")
>>> ts1.TRADING_DAYS_PER_YEAR == ts2.TRADING_DAYS_PER_YEAR
True  # Même valeur, partagée
>>> ts1.values == ts2.values
False  # Valeurs différentes, propres à chaque instance
```

---

### Concept fondamental : Type Hints (Annotations de type)

Les **type hints** sont des annotations qui indiquent le type attendu des variables, paramètres et valeurs de retour :

```python
def __init__(self, values: list[float], name: str = "unnamed") -> None:
#                  ↑               ↑              ↑              ↑
#            paramètre      type attendu    valeur défaut   type retour
```

**Points importants :**

1. **Pas d'effet à l'exécution** : Python n'applique pas les types au runtime. Ce code fonctionne même si on passe des types incorrects.

2. **Vérification statique** : Les outils comme `mypy`, `pyright` ou l'IDE (VS Code/PyCharm) analysent les types avant l'exécution.

3. **Documentation vivante** : Les types rendent le code plus lisible et auto-documenté.

```python
# Ce code s'exécute sans erreur malgré les types incorrects
ts = PriceSeries("pas une liste", 12345)  # ⚠️ Aucune erreur Python !
# Mais un type checker signalerait le problème
```

**Principe de Postel** (utile pour le typage) :
> "Soyez conservateur dans ce que vous envoyez, libéral dans ce que vous acceptez."

**Types courants :**

```python
# Types simples
x: int = 5
y: float = 3.14
s: str = "hello"
b: bool = True

# Types composés
numbers: list[float] = [1.0, 2.0]
mapping: dict[str, int] = {"a": 1}
optional: str | None = None  # Python 3.10+
# ou: Optional[str] = None   # versions antérieures

# Type de retour
def compute() -> float:
    return 3.14

def no_return() -> None:
    print("Je ne retourne rien")
```

---

### Étape 1.5 : Les méthodes `__repr__` et `__str__`

Ces méthodes spéciales définissent comment votre objet est représenté sous forme de chaîne de caractères.

```python
class PriceSeries:
    # ... (code précédent)
    
    def __repr__(self) -> str:
        """Représentation pour les développeurs (debugging)."""
        return f"PriceSeries({self.name!r}, {len(self.values)} points)"
    
    def __str__(self) -> str:
        """Représentation pour les utilisateurs."""
        if self.values:
            return f"{self.name}: {self.values[-1]:.2f} (latest)"
        return f"{self.name}: empty"
```

**Différence clé :**

| Méthode | Appelée par | Usage | Format |
|---------|-------------|-------|--------|
| `__repr__` | `repr(obj)`, REPL, debugger | Développement | Doit ressembler au code pour recréer l'objet |
| `__str__` | `str(obj)`, `print(obj)` | Utilisateur final | Lisible et convivial |

```python
>>> ts = PriceSeries([100.0, 102.5, 105.0], "AAPL")
>>> ts                    # REPL appelle __repr__
PriceSeries('AAPL', 3 points)
>>> print(ts)             # print() appelle __str__
AAPL: 105.00 (latest)
>>> repr(ts)
"PriceSeries('AAPL', 3 points)"
>>> str(ts)
'AAPL: 105.00 (latest)'
```

> **Astuce `!r` dans les f-strings** : L'expression `{self.name!r}` applique `repr()` à la valeur, affichant les guillemets pour les strings. Cela aide à distinguer `PriceSeries('AAPL', ...)` de `PriceSeries(AAPL, ...)`.

> **Conseil de Fluent Python** : Si vous n'implémentez qu'une seule méthode, choisissez `__repr__`. La classe `object` fournit un `__str__` par défaut qui appelle `__repr__` si `__str__` n'est pas défini.

---

### Concept fondamental : Le Python Data Model et les méthodes spéciales

D'après *Fluent Python* de Luciano Ramalho :

> "Le Python Data Model décrit l'API que nous utilisons pour que nos propres objets interagissent avec les fonctionnalités les plus idiomatiques du langage."

Les **méthodes spéciales** (ou "dunder methods" pour "double underscore") permettent à vos objets de s'intégrer naturellement avec les opérations Python :

```python
# Ce que vous écrivez          # Ce que Python appelle
len(obj)                       obj.__len__()
str(obj)                       obj.__str__()
repr(obj)                      obj.__repr__()
obj[key]                       obj.__getitem__(key)
obj1 + obj2                    obj1.__add__(obj2)
for x in obj:                  obj.__iter__()
```

**Règle importante** : Vous n'appelez généralement pas ces méthodes directement. C'est l'interpréteur Python qui les appelle pour vous. Écrivez `len(ts)`, pas `ts.__len__()`.

---

### Étape 1.6 : Premier commit Git

Sauvegardons notre progression :

```bash
# Voir les fichiers modifiés
git status

# Ajouter les fichiers
git add src/pyvest/priceseries.py

# Créer un commit avec un message descriptif
git commit -m "feat(core): add PriceSeries class with __init__, __repr__, __str__"

# Vérifier l'historique
git log --oneline
```

**Convention de commit** : Utilisez des messages clairs suivant le format : `type(scope): description`
- `feat`: nouvelle fonctionnalité
- `fix`: correction de bug
- `docs`: documentation
- `refactor`: refactorisation sans changement fonctionnel

---

### Étape 1.7 : Implémentation des méthodes de calcul de rendement

```python
import math

class PriceSeries:
    # ... (code précédent)
    
    def linear_return(self, t: int) -> float:
        """Rendement linéaire (arithmétique) entre t-1 et t."""
        return (self.values[t] - self.values[t-1]) / self.values[t-1]
    
    def log_return(self, t: int) -> float:
        """Log-rendement entre t-1 et t."""
        return math.log(self.values[t] / self.values[t-1])
```

**Test dans le REPL :**

```python
>>> ts = PriceSeries([100.0, 105.0, 103.0, 110.0], "TEST")
>>> ts.linear_return(1)  # (105 - 100) / 100
0.05
>>> ts.log_return(1)     # ln(105 / 100)
0.04879...

# Vérification de l'additivité des log-rendements
>>> sum(ts.log_return(t) for t in range(1, len(ts.values)))
0.09531...
>>> math.log(110 / 100)  # Log du ratio total
0.09531...  # Identique !
```

**Propriétés des rendements :**

| Type | Additivité | Usage principal |
|------|------------|-----------------|
| **Linéaire (arithmétique)** | Entre actifs : `r_p = Σ(w_i × r_i)` | Portefeuille, cross-section |
| **Logarithmique** | Dans le temps : `r_total = Σ(r_t)` | Série temporelle, volatilité |

**Approximation pour petits rendements** (développement de Taylor) :

```
ln(1 + r) ≈ r    pour |r| < 5%

Développement complet : ln(1+r) = r - r²/2 + r³/3 - r⁴/4 + ...
Erreur de troncature ≈ r²/2
Exemple : r = 5% → erreur ≈ 0.05²/2 = 0.00125 = 0.125%
```

---

### Concept fondamental : Les fonctions en Python

Une **fonction** est un bloc de code réutilisable qui effectue une tâche spécifique.

```python
def nom_de_fonction(param1: type1, param2: type2 = valeur_defaut) -> type_retour:
    """Docstring décrivant la fonction."""
    # Corps de la fonction
    resultat = param1 + param2
    return resultat
```

**Composants :**

- `def` : mot-clé introduisant la définition
- `nom_de_fonction` : identifiant (convention : `snake_case`)
- `param1, param2` : paramètres recevant des valeurs à l'appel
- `-> type_retour` : annotation du type de retour
- `return` : instruction (optionnelle) renvoyant une valeur

**Appel de fonction :**

```python
# Appel positionnel
resultat = nom_de_fonction(10, 20)

# Appel avec arguments nommés (ordre modifiable)
resultat = nom_de_fonction(param2=20, param1=10)

# Mix des deux (positionnels d'abord)
resultat = nom_de_fonction(10, param2=20)
```

---

### Concept fondamental : Portée des variables (Scope) - Règle LEGB

Python résout les noms de variables selon la règle **LEGB** :

```
L - Local      : Variables définies dans la fonction courante
E - Enclosing  : Variables des fonctions englobantes (closures)
G - Global     : Variables au niveau du module
B - Built-in   : Fonctions et constantes intégrées (len, print, True, ...)
```

```python
# GLOBAL
x = "global"

def externe():
    # ENCLOSING (pour interne)
    y = "enclosing"
    
    def interne():
        # LOCAL
        z = "local"
        print(z)  # → "local"
        print(y)  # → "enclosing" (trouvé dans Enclosing)
        print(x)  # → "global" (trouvé dans Global)
        print(len)  # → <built-in function len> (Built-in)
    
    interne()

externe()
```

**Piège classique** - Modifier une variable globale :

```python
compteur = 0

def incrementer():
    compteur = compteur + 1  # ❌ UnboundLocalError!
    # Python crée une variable locale 'compteur' qui n'est pas encore définie

def incrementer_correct():
    global compteur  # Déclare qu'on utilise la variable globale
    compteur = compteur + 1  # ✓ Fonctionne
```

---

### Différence entre fonction Python et fonction mathématique

| Aspect | Fonction mathématique | Fonction Python |
|--------|----------------------|-----------------|
| **Mapping** | Un élément de A → un unique élément de B | Peut ne rien retourner (`None`) |
| **Déterminisme** | Même input → même output (toujours) | Peut dépendre d'état externe |
| **Effets de bord** | Aucun | Peut modifier l'environnement |
| **Imbrication** | Non applicable | Fonctions imbriquées (closures) |

```python
import random

# Cette fonction n'est PAS une fonction mathématique pure
def generer_prix() -> float:
    return random.uniform(90, 110)  # Résultat différent à chaque appel

# Celle-ci est plus proche d'une fonction mathématique
def calculer_rendement(p0: float, p1: float) -> float:
    return (p1 - p0) / p0  # Déterministe, pas d'effet de bord
```

---

### Étape 1.8 : La méthode `__len__`

```python
class PriceSeries:
    # ... (code précédent)
    
    def __len__(self) -> int:
        """Retourne le nombre de points de données."""
        return len(self.values)
```

Cette méthode permet à nos instances de fonctionner avec la fonction built-in `len()` :

```python
>>> ts = PriceSeries([100.0, 105.0, 103.0, 110.0], "TEST")
>>> len(ts)
4
>>> bool(ts)  # __len__ active aussi le test de vérité
True
>>> empty = PriceSeries([], "EMPTY")
>>> len(empty)
0
>>> bool(empty)  # Une longueur de 0 est "falsy"
False
```

> **Duck Typing en action** : Python ne vérifie pas le type ; il vérifie la présence de la méthode. Tout objet avec `__len__` fonctionne avec `len()`.

---

### Concept fondamental : Duck Typing

> "If it walks like a duck and quacks like a duck, it's a duck."
> — Proverbe pythonique

Le **duck typing** signifie que Python se soucie des **comportements** (méthodes/attributs) plutôt que des types. Un objet est compatible avec une opération s'il implémente les méthodes nécessaires.

```python
def afficher_longueur(obj):
    """Fonctionne avec TOUT objet ayant __len__."""
    print(f"Longueur: {len(obj)}")

# Tous ces appels fonctionnent !
afficher_longueur([1, 2, 3])           # list
afficher_longueur("hello")              # str
afficher_longueur({"a": 1, "b": 2})     # dict
afficher_longueur(PriceSeries([100, 101, 102], "X"))  # notre classe !
```

C'est pourquoi implémenter `__len__` permet à `PriceSeries` de s'intégrer naturellement avec tout code utilisant `len()`.

---

### Étape 1.9 : Le décorateur `@property`

```python
class PriceSeries:
    # ... (code précédent)
    
    @property
    def total_return(self) -> float:
        """Rendement total sur toute la période."""
        if len(self.values) < 2:
            return 0.0
        return (self.values[-1] - self.values[0]) / self.values[0]
```

Le décorateur `@property` transforme une méthode en **attribut calculé** :

```python
>>> ts = PriceSeries([100.0, 105.0, 103.0, 110.0], "TEST")
>>> ts.total_return  # Pas de parenthèses !
0.1
>>> f"Rendement total: {ts.total_return:.2%}"
'Rendement total: 10.00%'
```

**Avantages de `@property` :**

1. **Syntaxe propre** : `ts.total_return` au lieu de `ts.total_return()`
2. **Encapsulation** : Le calcul est caché derrière une interface simple
3. **Lazy evaluation** : Calculé uniquement quand on y accède

**Quand utiliser `@property` :**

- Valeur dérivée d'autres attributs
- Pas d'effets de bord
- Calcul relativement léger

---

### Concept fondamental : Les variables ne sont pas des boîtes

D'après *Fluent Python* :

> "L'habituelle métaphore des 'variables comme boîtes' entrave la compréhension des variables de référence. [...] Pensez aux variables comme des étiquettes attachées aux objets."

```python
# ❌ Mauvaise intuition (boîtes)
# On pourrait croire que b contient une COPIE de a

# ✓ Bonne intuition (étiquettes)
# a et b sont des étiquettes pointant vers le MÊME objet

a = [1, 2, 3]
b = a           # b est une autre étiquette sur le même objet
a.append(4)
print(b)        # [1, 2, 3, 4] - b voit le changement !
```

**Illustration :**

```markdown
Mauvaise intuition:          Bonne intuition:
┌───┐     ┌───┐              ┌─────────────┐
│ a │     │ b │              │ [1, 2, 3, 4]│
│[1,│     │[1,│         a ───►             ◄─── b
│2, │     │2, │              └─────────────┘
│3] │     │3] │              (même objet en mémoire)
└───┘     └───┘
 (copie?)  (copie?)
```

---

### Concept fondamental : Objets mutables vs immutables

| Type | Mutabilité | Exemples |
|------|------------|----------|
| **Immutable** | Ne peut pas être modifié après création | `int`, `float`, `str`, `tuple`, `frozenset` |
| **Mutable** | Peut être modifié après création | `list`, `dict`, `set`, objets personnalisés |

```python
# Immutable - l'objet original n'est pas modifié
x = 5
y = x
x = x + 1  # Crée un NOUVEL objet int
print(y)   # 5 - y pointe toujours vers l'ancien objet

s = "hello"
s.upper()  # Retourne "HELLO" mais ne modifie pas s
print(s)   # "hello" - inchangé

# Mutable - l'objet original EST modifié
a = [1, 2, 3]
b = a
a.append(4)  # Modifie l'objet existant
print(b)     # [1, 2, 3, 4] - b voit le changement !
```

**Implication pour `PriceSeries` :**

```python
# ⚠️ Danger potentiel
prices = [100, 105, 110]
ts = PriceSeries(prices, "TEST")
prices.append(115)  # Modifie aussi ts.values !
print(ts.values)    # [100, 105, 110, 115]

# ✓ Solution : copie défensive dans __init__
def __init__(self, values: list[float], name: str = "unnamed") -> None:
    self.values = list(values)  # Crée une COPIE de la liste
    self.name = name
```

---

### Concept fondamental : `is` vs `==`

| Opérateur | Compare | Question posée |
|-----------|---------|----------------|
| `==` | Les **valeurs** | "Ont-ils le même contenu ?" |
| `is` | L'**identité** (adresse mémoire) | "Sont-ils le même objet ?" |

```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

a == b   # True  - mêmes valeurs
a is b   # False - objets différents
a == c   # True  - mêmes valeurs
a is c   # True  - même objet

# Vérification avec id()
id(a)    # 140234567890
id(b)    # 140234567891 - différent !
id(c)    # 140234567890 - identique à a
```

**Règle pour `None` :**

```python
x = None

# ✓ Correct - toujours utiliser 'is' pour None
if x is None:
    print("x est None")

# ⚠️ Fonctionne mais déconseillé
if x == None:
    print("x est None")
```

> **Pourquoi ?** `None` est un singleton en Python (un seul objet `None` existe). Utiliser `is` est plus rapide et plus explicite.

---

### Concept fondamental : Slicing (découpage)

Le slicing permet d'extraire des sous-séquences avec la syntaxe `sequence[start:stop:step]` :

```python
prices = [100, 105, 103, 110, 108, 112]
#          0    1    2    3    4    5   (indices positifs)
#         -6   -5   -4   -3   -2   -1   (indices négatifs)

# Extraction de base
prices[0]       # 100 - premier élément
prices[-1]      # 112 - dernier élément
prices[1:4]     # [105, 103, 110] - indices 1, 2, 3 (stop exclu)
prices[:3]      # [100, 105, 103] - du début à l'indice 2
prices[3:]      # [110, 108, 112] - de l'indice 3 à la fin
prices[::2]     # [100, 103, 108] - un élément sur deux
prices[::-1]    # [112, 108, 110, 103, 105, 100] - inversé

# Utilisé dans notre code
peak = max(self.values[:t+1])  # Maximum du début jusqu'à t (inclus)
last_price = self.values[-1]   # Dernier prix
```

---

### Concept fondamental : Gestion des erreurs (Exceptions)

Les **exceptions** signalent des conditions anormales. Sans gestion, elles arrêtent le programme.

```python
# ❌ Code fragile - peut planter
def linear_return(self, t: int) -> float:
    return (self.values[t] - self.values[t-1]) / self.values[t-1]
    # Que se passe-t-il si t=0 ? → values[-1] (dernier élément, pas ce qu'on veut!)
    # Que se passe-t-il si t=100 et len(values)=50 ? → IndexError

# ✓ Code robuste avec validation
def linear_return(self, t: int) -> float:
    """Rendement linéaire entre t-1 et t."""
    if t < 1:
        raise ValueError(f"t doit être >= 1, reçu: {t}")
    if t >= len(self.values):
        raise IndexError(f"t={t} hors limites (max: {len(self.values)-1})")
    return (self.values[t] - self.values[t-1]) / self.values[t-1]
```

**Gestion avec `try/except` :**

```python
try:
    ret = ts.linear_return(0)
except ValueError as e:
    print(f"Erreur de valeur: {e}")
except IndexError as e:
    print(f"Index hors limites: {e}")
```

**Exceptions courantes :**

| Exception | Cause typique |
|-----------|---------------|
| `ValueError` | Valeur incorrecte mais type correct |
| `TypeError` | Type incorrect |
| `IndexError` | Index hors limites d'une séquence |
| `KeyError` | Clé absente d'un dictionnaire |
| `ZeroDivisionError` | Division par zéro |
| `FileNotFoundError` | Fichier introuvable |

---

### Étape 1.10 : Deuxième commit

```bash
git add .
git commit -m "feat(core): add return calculations and __len__ to PriceSeries"
```

---

## Partie 2 : Exercices Pratiques

---

### Exercice 1 : Calcul du vecteur de rendements (15 min)

Implémentez deux méthodes retournant la liste de tous les rendements :

```python
def all_linear_returns(self) -> list[float]:
    """Retourne la liste de tous les rendements linéaires.
    
    Returns:
        Liste de n-1 rendements pour n prix.
    """
    # Votre code ici
    pass

def all_log_returns(self) -> list[float]:
    """Retourne la liste de tous les log-rendements."""
    # Votre code ici
    pass
```

**Test attendu :**

```python
>>> ts = PriceSeries([100, 105, 110], "TEST")
>>> ts.all_linear_returns()
[0.05, 0.047619...]
>>> ts.all_log_returns()
[0.04879..., 0.04652...]
```

---

### Exercice 2 : Volatilité annualisée (25 min)

```python
def annualized_volatility(self) -> float:
    """
    Volatilité annualisée à partir des log-rendements.
    
    Formule: σ_annual = σ_daily × √252
    
    Note: Le scaling √252 suppose des rendements IID 
    (indépendants et identiquement distribués).
    Cette hypothèse est rarement vérifiée en pratique 
    (clustering de volatilité).
    
    Pour une meilleure estimation, considérer:
    - Modèles GARCH
    - Moyenne mobile exponentielle (EWMA)
    - Volatilité réalisée (données haute fréquence)
    """
    # Étapes:
    # 1. Obtenir tous les log-rendements
    # 2. Calculer la moyenne
    # 3. Calculer la variance d'échantillon (diviser par n-1)
    # 4. Prendre la racine carrée pour la volatilité quotidienne
    # 5. Annualiser: × √252
    pass
```

**Indications :**
- Utiliser `math.sqrt()`
- Utiliser `(n-1)` pour l'estimateur non biaisé de la variance
- Retourner `0.0` si moins de 3 prix

---

### Exercice 3 : Ratio de Sharpe (20 min)

```python
def annualized_return(self) -> float:
    """
    Rendement annuel moyen à partir des log-rendements.
    Formule: μ_annual = μ_daily × 252
    """
    pass

def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
    """
    Ratio de Sharpe annualisé.
    
    Formule: SR = (μ - r_f) / σ
    
    D'après "Elements of Quantitative Investing":
    "Le ratio de Sharpe est le rendement par unité de risque."
    
    Args:
        risk_free_rate: Taux sans risque ANNUEL (défaut: 0)
    
    Returns:
        Ratio de Sharpe (sans dimension)
    
    Note:
        Le ratio de Sharpe scale avec √T où T est l'horizon.
        SR_annual ≈ SR_daily × √252
    """
    pass
```

---

### Exercice 4 : Analyse du Drawdown (25 min)

```python
def drawdown_at(self, t: int) -> float:
    """
    Calcule le drawdown au temps t.
    
    Drawdown = (prix_actuel - pic) / pic
    Le pic est le prix maximum du début jusqu'au temps t.
    
    Args:
        t: Index temporel
    
    Returns:
        Drawdown en décimal négatif (-0.10 = baisse de 10%)
    """
    pass

def max_drawdown(self) -> float:
    """
    Drawdown maximum sur toute la série.
    
    Returns:
        Drawdown maximum (valeur négative ou zéro)
    """
    pass
```

**Test attendu :**

```python
>>> ts = PriceSeries([100, 110, 105, 95, 100, 90], "TEST")
>>> ts.drawdown_at(2)   # 105 vs pic 110
-0.0454...
>>> ts.drawdown_at(5)   # 90 vs pic 110
-0.1818...
>>> ts.max_drawdown()
-0.1818...
```

---

### Exercice 5 : Intégration finale (20 min)

```python
def summary(self) -> str:
    """Génère un rapport de performance."""
    return f"""
    Performance Summary: {self.name}
    ══════════════════════════════════
    Points de données: {len(self)}
    Premier prix:      ${self.values[0]:.2f}
    Dernier prix:      ${self.values[-1]:.2f}
    Rendement total:   {self.total_return:.2%}
    Volatilité:        {self.annualized_volatility():.2%}
    Ratio de Sharpe:   {self.sharpe_ratio():.2f}
    Max Drawdown:      {self.max_drawdown():.2%}
    """
```

---

### Commit final

```bash
git add .
git commit -m "feat(core): add volatility, sharpe, drawdown to PriceSeries"
```

---

## Solutions

### Solution - Exercice 1

```python
def all_linear_returns(self) -> list[float]:
    """Retourne la liste de tous les rendements linéaires."""
    return [self.linear_return(t) for t in range(1, len(self.values))]

def all_log_returns(self) -> list[float]:
    """Retourne la liste de tous les log-rendements."""
    return [self.log_return(t) for t in range(1, len(self.values))]
```

---

### Solution - Exercice 2

```python
def annualized_volatility(self) -> float:
    """Volatilité annualisée à partir des log-rendements."""
    if len(self.values) < 3:
        return 0.0
    
    log_rets = self.all_log_returns()
    n = len(log_rets)
    mean = sum(log_rets) / n
    variance = sum((r - mean) ** 2 for r in log_rets) / (n - 1)  # Bessel's correction
    daily_vol = math.sqrt(variance)
    
    return daily_vol * math.sqrt(self.TRADING_DAYS_PER_YEAR)
```

---

### Solution - Exercice 3

```python
def annualized_return(self) -> float:
    """Rendement annuel moyen à partir des log-rendements."""
    if len(self.values) < 2:
        return 0.0
    log_rets = self.all_log_returns()
    mean_daily = sum(log_rets) / len(log_rets)
    return mean_daily * self.TRADING_DAYS_PER_YEAR

def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
    """Ratio de Sharpe annualisé."""
    vol = self.annualized_volatility()
    if vol == 0:
        return 0.0
    excess_return = self.annualized_return() - risk_free_rate
    return excess_return / vol
```

---

### Solution - Exercice 4

```python
def drawdown_at(self, t: int) -> float:
    """Drawdown au temps t."""
    peak = max(self.values[:t+1])
    return (self.values[t] - peak) / peak

def max_drawdown(self) -> float:
    """Drawdown maximum sur toute la série."""
    if len(self.values) < 2:
        return 0.0
    return min(self.drawdown_at(t) for t in range(len(self.values)))
```

---

## Résumé de la Session 1

### Concepts Python appris

| Concept | Application |
|---------|-------------|
| Définition de `class` | Template pour `PriceSeries` |
| `__init__` | Initialisation avec prix et nom |
| `__repr__` / `__str__` | Représentations string |
| `__len__` | Intégration avec `len()` |
| `@property` | Attributs calculés |
| Type hints | Documentation du code |
| Attributs de classe | Constantes partagées (`TRADING_DAYS_PER_YEAR`) |
| Attributs d'instance | Données propres (`values`, `name`) |
| Slicing | Extraction de sous-séquences |
| Exceptions | Gestion des erreurs |

### Concepts de finance quantitative

- Rendements linéaires vs logarithmiques
- Additivité des rendements
- Volatilité et ratio de Sharpe
- Analyse du drawdown

---

## Travail à faire

1. **Ajouter une méthode** `rolling_mean(window: int)` calculant la moyenne mobile des rendements
2. **Recherche** : Quelle est la différence entre l'écart-type d'échantillon et l'écart-type de population ?
3. **Lecture** : Chapitre 1 de *Fluent Python* sur le Python Data Model

---

## Aperçu de la Session 2

Dans la prochaine session, nous :

- Construirons une classe `Asset` utilisant `PriceSeries` (composition)
- Créerons un `DataLoader` pour récupérer des données réelles
- Apprendrons les collections Python (listes, dictionnaires, ensembles)
- Traiterons plusieurs actifs efficacement