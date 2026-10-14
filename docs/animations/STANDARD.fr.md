# Norme pour une vidéo pédagogique Bioptim

*English version: [STANDARD.md](STANDARD.md).*

La spécification de référence des vidéos de `docs/animations/`. Une nouvelle vidéo, réalisée par une personne ou par un
agent, doit la suivre pour être indiscernable des 48 existantes. Tout ce qui suit est vérifiable ; la section 7 est la
liste de contrôle à cocher avant d'accepter une vidéo. Partez de [templates/scene_template.py](templates/scene_template.py)
(un squelette fonctionnel qui suit toutes les règles) et consultez [SERIES_LAYER.md](SERIES_LAYER.md) pour ce que la
couche de rendu fait à votre place.

Quand les anciens documents contredisent ce fichier (voir « Décisions » à la fin de la section 7), ce fichier prévaut.

## 1. Objectif, public, périmètre

* **Public** : étudiants et chercheurs en biomécanique et en commande optimale qui connaissent un peu Python et la
  mécanique, pas forcément le tir multiple direct. La vidéo est une porte d'entrée vers l'exemple et le code Bioptim, pas
  un remplacement.
* **Une idée par vidéo.** Le spectateur doit pouvoir dire en une phrase ce qu'il a appris (cette phrase est la
  `description_en` de l'entrée du catalogue). S'il faut « et aussi », faites deux vidéos.
* **Durée** : 10 à 20 s de contenu à la vitesse native de Manim. La couche multiplie chaque `run_time` par 1,6 (plus des
  pauses de lecture de 0,5 s) et ajoute une carte de fin de 4,5 s : la vidéo finale dure environ 16 à 32 s + 4,5 s. Au plus
  **2 temps** (un temps est une séquence autonome, par ex. « la grille de nœuds » puis « de vraies résolutions » ; le second
  commence en effaçant tout sauf le titre). Les scènes les plus anciennes sont plus longues (médiane 31 s au total) : ne
  les prenez pas comme référence de durée.
* **Langue** : les scènes sont écrites en anglais ; le français est produit par la couche à partir de
  `i18n/fr_<sujet>.json` (section 5).
* **Niveau** (catalogue) : 1 introduction, 2 intermédiaire, 3 avancé (suppose des vidéos antérieures).

## 2. Règles d'honnêteté des données

1. **Chaque courbe et chaque nombre provient d'une VRAIE résolution bioptim** (ou d'un vrai objet de la bibliothèque :
   un vrai `OdeSolver`, `VectorLayout`, `Solution`...), produite par `generate_<sujet>_data.py` et stockée dans
   `data/<sujet>_*.npz` (petit : de quelques ko à quelques dizaines de ko). Rien n'est tapé à la main, extrapolé, lissé ou
   « esquissé ».
2. **Les données d'entrée synthétiques** (mesures simulées, marqueurs bruités, référence inventée) ne sont admises que si la
   scène le dit à l'écran (sous-titre ou légende : « données synthétiques : mouvement simulé + bruit gaussien (sigma =
   0,05 rad) »), avec le niveau de bruit et la graine dans le générateur et les notes.
3. **Les interfaces redessinées** (sortie IPOPT, fenêtre de tracé, tableau d'un objet Bioptim) sont étiquetées « redessiné,
   pas une capture d'écran » à l'écran, et leur contenu est réel.
4. **Rapporter honnêtement le solveur** : le statut IPOPT et le nombre d'itérations sont stockés (`status`, `iterations`,
   `converged`) et affichés (`ipopt_line`). Dire à l'écran ou dans les notes quand une résolution est initialisée à chaud ou
   utilise une continuation (chaque résolution part de la précédente), et quand plusieurs minima locaux existent (un
   problème non convexe renvoie *un* minimum local). Ne jamais présenter comme résultat une solution non convergée ; si
   l'échec est le sujet (par ex. `CONTINUOUS` au lieu de `IMPACT` est infaisable), l'étiqueter (« dernière itération
   d'IPOPT, pas une solution »).
5. **Si le sujet ne peut pas être mis en œuvre** (pas de convergence, une API qui ne fait pas ce que dit la documentation
   dans cette version), arrêtez-vous, gardez le script qui échoue et signalez-le. N'inventez pas un résultat et ne changez
   pas l'énoncé pour l'accorder aux chiffres.
6. **Les nombres cités à l'écran sont calculés à partir du npz** dans la scène (`float(np.abs(tau).max())`, puis
   `f"{pic:.1f}"`), jamais lus à l'œil sur un tracé ni recopiés en littéral dans la chaîne. Il en va de même des notes : elles
   citent les valeurs affichées par le générateur.
7. **Le code montré est le code qui a tourné.** Chaque ligne du panneau de code est (une simplification d')une ligne de
   `generate_<sujet>_data.py`, et chaque identifiant, argument nommé et valeur par défaut est vérifié dans les sources de
   CETTE version (`grep -n "def nom\|class Nom" bioptim/...`), pas dans la documentation ni de mémoire. Un appel simplifié
   (bornes ou imports omis, `...`) n'est admis que si l'exemple lié fait référence et que les notes le disent.
8. **Les simplifications sont listées** dans les notes (« Réserves honnêtes ») : modèle simplifié, durées fixées, contraintes
   absentes, choix de N, T, tolérances.

## 3. Anatomie visuelle (1920x1080, cadre Manim de 14,2 x 8 unités)

```
+--------------------------------------------------------------------------------+
|                    TITRE (34 gras)  /  sous-titre (22 gris)                    |  y = +3,7
| étiquette d'axe (20)                         Bioptim code   (légende, 20 gris)  |
|  +-----------------------+                   objectives.add(...   (code, 17-20)|  CODE_X = 0,15
|  |  tracé 1 (fantôme en  |                   ...                              |  CODE_W = 6,75
|  |  pointillés + courbe) |                                                     |
|  +-----------------------+                   bloc de valeurs : poids = 10 ...  |
|  |  tracé 2 : différence |                   IPOPT : 22 itérations, convergé   |
|  +-----------------------+   t (s)           remarque (jaune, 17-19)           |
| phrase de bas de page (16, 2 lignes max., s'arrête avant le logo)      [logo]  |  y = -3,9
+--------------------------------------------------------------------------------+
```

(Le diagramme montre les textes anglais, tels qu'ils sont écrits dans les scènes.)

* **Titre** : `scene_title(titre, sous-titre)` : titre 34 gras, sous-titre 22 gris (`GRAY_B`), centrés en haut. Le titre
  nomme le concept (« Penalty on the derivative of a control »), le sous-titre donne le cadre avec ses nombres
  (« swing-up, N = 30, T = 1 s »). Les données synthétiques sont déclarées ici.
* **Gauche = visuels, droite = code.** Les tracés vont de `PLOT_X0 = -6,6` à `PLOT_X1 = -0,5` en x (`make_axes` centré en
  x = -3,55, largeur 5,6) ; le panneau de code commence à `CODE_X = 0,15`, mesure au plus `CODE_W = 6,75`, son haut est à
  y = 2,3 ; les autres textes de droite mesurent au plus `TEXT_W = 5,9` (`place`).
* **Légende « Bioptim code »** : tout panneau de code porte la légende `Bioptim code` (taille 20, `GRAY_B`) **au-dessus**
  des lignes de code. Utilisez `code_panel(lignes)` de `features_scenes.py` : il ajoute la légende, décale de 0,3 unité par
  niveau les lignes `(niveau, texte, couleur)` et met à l'échelle `CODE_W`. Si une légende plus longue est nécessaire,
  gardez le début (`Bioptim code (M = walk_hopper.bioMod)`). Les lignes de code sont dans la police de code, taille 17-20
  (jamais moins de 15), un appel par ligne. Colorez les arguments qui changent et animez le changement (`Transform` de la
  ligne).
* **Bloc de valeurs** (à droite, sous le code) : nombres calculés à partir des données, taille 19, `GRAY_A`,
  `line_spacing=0.9`, construit comme UN seul `Text` avec retours à la ligne : `weight = 10 · peak |y| = 0.67` /
  `IPOPT: 22 iterations, converged` (`ipopt_line(iterations, converged)`). Placez-le avec `place(mob, CODE_X, y)`.
* **Remarque** (facultative, à droite) : `say(phrase, taille 17-19)` en `YELLOW_C` : une phrase entière, uniquement à propos
  de ce qui est tracé. Le retour à la ligne est fait par `para`.
* **Bas de page** : `footer(phrase)`, taille 16, `YELLOW_C` (ou `GRAY_B` pour une simple note), en bas à gauche
  (x = -6,9, y = -3,92) : une phrase, deux lignes au plus, largeur au plus 10,4 pour finir avant le logo en bas à droite.
  Servez-vous-en pour la réserve honnête de la vidéo (« Grille schématique avec N = 10 ; le vrai problème utilise N = 30 »).
* **Fantôme de référence et axe de différence.** Quand une grandeur *change* (un poids, une borne, un solveur, un modèle),
  rendez le changement visible : gardez la courbe de référence à l'écran en fantôme gris pointillé
  (`DashedVMobject(poly(...), num_dashes=40).set_opacity(0.8)`, couleur `GRAY_B`) et ajoutez un second axe, plus petit, avec
  la différence `nouveau - référence` (couleur `PURPLE_B`, ligne de zéro tracée), avec sa propre légende (« Δθ = θ − θ_free
  (rad) »). Ne racontez jamais ce qui n'est pas tracé : si le texte dit « le pic diminue », le pic doit être visible et le
  nombre dans le bloc de valeurs.
* **Rôles des couleurs** (constantes de `features_scenes.py` ; réutilisez-les, n'en ajoutez que pour un nouveau rôle, nommées
  `C_<ROLE>`) :

  | Rôle | Constante | Couleur |
  | --- | --- | --- |
  | commandes, Lagrange | `C_CTRL`, `C_LAG` | `GREEN_C` |
  | états, courbe qui change | `C_STATE` | `YELLOW_C` |
  | Mayer, seconde phase | `C_MAY`, `C_PH1` | `ORANGE` |
  | première phase | `C_PH0` | `BLUE_C` |
  | bornes, zone interdite, échec | `C_BOUND` | `RED_C` (bande : `band(...)`, opacité 0,22) |
  | temps, durée | `C_TIME` | `TEAL_C` |
  | paramètres, différences | `C_PAR` | `PURPLE_B` |
  | fantôme, étiquettes d'axe, texte secondaire | - | `GRAY_B` |
  | valeurs affichées | - | `GRAY_A` |
  | remarques, bas de page | - | `YELLOW_C` |

  Une courbe garde la même couleur dans tous les tracés et dans les lignes de code qui la créent.
* **Polices** : `FONT` (Segoe UI sous Windows, DejaVu Sans ailleurs) pour tout le texte, `MONO` (Consolas / DejaVu Sans
  Mono) pour le code ; les deux sont réglées par défaut à l'import de `features_scenes`. Tailles : titre 34, sous-titre 22,
  étiquettes d'axe et légende 20, valeurs 19, code 17-20, remarque 17-19, bas de page 16, graduations et `t (s)` 16, plus
  petite annotation 14.
* **Axes et unités** : `make_axes` (sans graduations tracées), puis `axis_label` (au-dessus de l'axe y, unité entre
  parenthèses : `angle (rad)`, `actuated force (N)`), `time_label(ax)` (`t (s)`), `x_ticks` / `y_ticks` pour les valeurs de
  graduation (elles appellent `dec()`). Une idée par étiquette d'axe ; le symbole d'abord si utile : `θ(t)  angle (rad)`.
* **Décimales** : les nombres formatés en Python dans une *phrase* restent simples (`f"{x:.2f}"`) ; la couche transforme le
  `.` en `,` en français. Les étiquettes purement numériques que vous construisez et qui ne contiennent aucun mot
  (graduations, valeurs de barres) passent par `dec(...)`, comme `x_ticks` / `y_ticks`. N'appliquez jamais `dec()` à du
  texte en police de code (un panneau de code français garde ses points).
* **Sécurité du cadre** : rien au-delà de x = ±7,1 ni y = ±4,0 (l'audit signale tout `Text` qui dépasse, avec une tolérance
  de 0,05), les textes ne se chevauchent pas (audit : recouvrement supérieur à 25 % du plus petit cadre), le coin du logo
  (automatique, 0,35 unité de haut, opacité 75 %, en bas à droite par défaut) reste libre. Le français est environ 20 % plus
  long : dessinez chaque texte de droite pour qu'il finisse encore avant x = 7,1 avec +20 % de largeur.
* **Fin** : la dernière instruction de `construct` est `self.wait(2.5)`. La couche ajoute ensuite la carte de fin.

## 4. Règles de texte

* **Des phrases entières** dans un seul objet `Text` (ou un `Text` multiligne fait par `para`/`say`, dont les retours à la
  ligne font partie de la clé de traduction). N'assemblez jamais une phrase avec plusieurs `Text` : l'ordre des mots
  diffère en français.
* **Les nombres dans la même chaîne** que la phrase, en nombre simple (`f"peak |τ| = {peak:.1f} N"`) : la couche transforme
  chaque nombre en `{0}`, `{1}` de la clé de traduction. Ne mettez pas un nombre dans un `Text` séparé à côté de ses mots.
* **Aucune abréviation sans développement** la première fois (`OCP` -> « problème de commande optimale (OCP) » ; `NLP`,
  `DMS`, `RK4` sont explicités dans le sous-titre ou le bas de page de la vidéo qui les introduit). Les symboles (`θ`, `τ`,
  `∫`) sont admis quand l'étiquette d'axe dit aussi le mot.
* **Transitions simples uniquement** : `FadeIn`, `FadeOut`, `Create` (tracés), `Transform` (courbes, nombres, lignes de
  code). Pas d'écriture lettre à lettre, ni `Wiggle`, ni `Indicate` sur du texte ; la couche les convertit de toute façon
  (SERIES_LAYER.md, section 3) mais la scène doit se lire bien sans.
* **Le code n'est jamais traduit** : texte en police de code, identifiants (`ObjectiveFcn.Lagrange.MINIMIZE_CONTROL`,
  `n_shooting`, `IPOPT`), noms de fichiers restent en anglais, y compris dans les phrases françaises. La légende
  `Bioptim code` est une clé normale (`Code Bioptim` en français).
* **Pas de `Paragraph`** (Manim coupe les lignes traduites au nombre de glyphes anglais ; la couche contourne le problème mais
  la règle évite le piège) : un `Text` par ligne (helper `Lines(...)` de `anim_mhe.py`, ou `say`/`para`).
* **Pas de `MathTex`/`Tex`/LaTeX** (non installé, non traduit). Écrivez les formules avec `Text`/`MarkupText` et Unicode
  (`∫ L(x, u) dt`, `t<sub>N−1</sub>` avec `MarkupText`).
* Pas de cartes `t2c`/`t2w` ni de découpage par indices sur du texte : ils se rapportent à la chaîne anglaise ; colorez plutôt
  des objets entiers.

## 5. Fichiers, nommage, catalogue, traduction

| Quoi | Où | Remarques |
| --- | --- | --- |
| Classes de scène | `docs/animations/anim_<sujet>.py` | `<sujet>` en minuscules, un fichier par vidéo ; nom de classe CamelCase = nom du mp4 |
| Générateur de données | `docs/animations/generate_<sujet>_data.py` | vraie résolution bioptim, affiche statut/itérations, écrit le npz ; la docstring explique ce qui est stocké ; ligne d'usage `PYTHONPATH=. python docs/animations/generate_<sujet>_data.py` |
| Données | `docs/animations/data/<sujet>_*.npz` | petites ; versionnées (seuls binaires admis) |
| Modèles | `docs/animations/models/<sujet>_*.bioMod` | seulement si aucun modèle de `bioptim/examples/models` ne convient |
| Notes | `docs/animations/notes/<sujet>.md` | canevas ci-dessous |
| Entrée de catalogue | `docs/animations/catalog.json` | champs ci-dessous |
| Traduction | `docs/animations/i18n/fr_<sujet>.json` | clé -> gabarit avec `{0}`, `{1}` |

Le fichier de scène commence par une docstring (ce qu'il montre, fichier de données, générateur, nom et durée de la scène,
commande de rendu), importe `from manim import *` puis les helpers de `features_scenes` (`scene_title`, `code_panel`,
`make_axes`, `poly`, `steps`, `axis_label`, `time_label`, `x_ticks`, `y_ticks`, `say`, `footer`, `place`, `ipopt_line`,
`hline`, `band`, `DATA_DIR`, constantes de couleur). Ne recopiez pas les helpers dans le fichier de scène ; s'il en manque
un, ajoutez-le à `features_scenes.py` dans un commit séparé. Formatez avec `black -t py311 -l120`.

**Fichier de notes** (`notes/<sujet>.md`), dans cet ordre :
1. `# Titre (anim_<sujet>.py)`, puis `## What the scene teaches` (ce que la scène enseigne) : l'idée, le cadre (modèle, N, T,
   solveur), les nombres montrés, pris dans la sortie du générateur ;
2. `## Commands` (commandes) : environnement, générateur, rendu (section 6) ;
3. `## Honest caveats` (réserves honnêtes) : minima locaux, initialisations à chaud, simplifications, temps de calcul, ce qui
   n'est *pas* montré ;
4. `## Exercises` (exercices) : 2 à 3 tâches numérotées et concrètes qui changent un argument du code montré.

Les notes existantes sont rédigées en anglais ; gardez les intitulés de section en anglais pour rester homogène.

**Entrée de catalogue** (tous les champs ; `id` vaut `<fichier>.py:<Classe>`) :

```json
{
  "id": "anim_<sujet>.py:MyScene", "slug": "my-scene",
  "title_en": "...", "title_fr": "...",
  "description_en": "une idée, le cadre, le résultat (avec des nombres)", "description_fr": "...",
  "video_basename": "MyScene", "duration_note": "",
  "links": [
    {"label_en": "Example: ...", "label_fr": "Exemple : ...", "path": "bioptim/examples/.../x.py", "lines": "40-90"},
    {"label_en": "Scene data generator", "label_fr": "Générateur des données de la scène",
     "path": "docs/animations/generate_<sujet>_data.py", "lines": "46-72"}
  ],
  "notes_file": "notes/<sujet>.md", "level": 2,
  "section": "Objectives and constraints", "section_fr": "Objectifs et contraintes"
}
```

* `section` est l'une des huit : Fundamentals (Fondamentaux), Discretization (Discrétisation), Objectives and constraints
  (Objectifs et contraintes), Phases and time (Phases et temps), Solvers and numerics (Solveurs et numérique), Models and
  biomechanics (Modèles et biomécanique), Advanced control (Commande avancée), Library overview (Vue d'ensemble de la
  bibliothèque). `section_fr` reprend exactement le nom français.
* `links` : cinq au plus (limite de la carte de fin), d'abord un **exemple** de `bioptim/examples/...`, puis des **lignes de
  bibliothèque**, enfin le générateur. Chaque `path` existe et chaque plage `lines` (`"40-90"`) est vérifiée en ouvrant le
  fichier de cette version et en s'assurant qu'elle couvre la définition nommée ; `build_readme_tables.py --check` ne
  vérifie que l'existence, pas les numéros de ligne.
* `logo_corner` (`tl`, `tr`, `bl`, `br`) seulement si le choix automatique entre encore en collision (audit json,
  `logo_overlap`).
* Ajoutez l'entrée avec `python docs/animations/render_series.py --gen-catalog` (ajoute un squelette pour chaque nouvelle
  `class X(Scene)` de `anim_*.py`, n'écrase jamais), puis remplissez tous les champs, dans les deux langues. Les gabarits de
  `templates/` ne sont pas explorés.
* Puis `python docs/animations/build_readme_tables.py` et mettez à jour les comptes écrits en toutes lettres dans `README.md` /
  `README.fr.md` (nombre de vidéos, statistiques de durée).

**Chaîne de traduction** (détails et glossaire dans [i18n/README.md](i18n/README.md)) :
1. `python docs/animations/render_series.py anim_<sujet>.py MyScene --dry --collect keys.jsonl` liste chaque clé
   (`fichier:ligne` d'origine) ;
2. écrire `i18n/fr_<sujet>.json` : la même clé, avec les nombres remplacés par `{0}`, `{1}` dans l'ordre d'apparition ; le
   gabarit peut les réordonner ; conserver, pour une clé multiligne, le même nombre de lignes ;
3. glossaire : OCP = problème de commande optimale, multiple shooting = tir multiple, direct collocation = collocation
   directe, node = nœud, controls = commandes, states = états, constraint = contrainte, bounds = bornes, cost = coût,
   warm start = initialisation à chaud, initial guess = estimation initiale, weight = poids. Les identifiants restent en
   anglais ;
4. `--lang fr --strict` doit réussir (code de sortie 3 = clé non traduite).

**Hygiène des commits** : seulement des fichiers sous `docs/animations/` (scène, générateur, npz, modèle, notes,
`catalog.json`, `i18n`, tableaux des README). Ne versionnez jamais `*.mp4`, `*.png`, `assets/bioptim_logo.png`, `media/`,
`*.jsonl`, `p.out` ni aucun fichier de sortie déposé à la racine du dépôt ; vérifiez `git status` avant de valider.

## 6. Commandes

Environnement de rendu (sans bioptim) : venv Python 3.11 avec `manim==0.21.*`, `numpy`, `black` ; polices Segoe UI et Consolas
sous Windows. Environnement des données : environnement conda avec bioptim, biorbd, casadi et IPOPT ; sous Windows, quand on
appelle son `python.exe` sans `conda activate` :

```bash
E=/c/Users/<vous>/miniconda3/envs/captury_biobuddy
export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Scripts:$PATH"   # sinon "Plugin 'ipopt' is not found"
export PYTHONIOENCODING=utf-8
```

Depuis la racine du dépôt :

```bash
PYTHONPATH=. python docs/animations/generate_<sujet>_data.py                      # vraie résolution -> data/<sujet>_*.npz
python docs/animations/render_series.py anim_<sujet>.py MyScene --dry              # construction seule (secondes)
python docs/animations/render_series.py anim_<sujet>.py MyScene --lang en          # 1080p30 (qualité par défaut)
python docs/animations/render_series.py anim_<sujet>.py MyScene --lang fr --strict # français, échoue si une clé manque
python docs/animations/render_series.py anim_<sujet>.py MyScene --lang both --quality 480p15   # aperçu rapide
python docs/animations/render_series.py --all --lang both --jobs 4 --out my_videos # toute la série
black -t py311 -l120 docs/animations/anim_<sujet>.py docs/animations/generate_<sujet>_data.py
python docs/animations/build_readme_tables.py                                      # tableaux des README + contrôle des liens
```

Sorties : `<out>/<Scene>_<lang>.mp4` (par défaut `docs/animations/media/series/out`, ignoré par git) et, dans `<out>/logs`,
`<Scene>_<lang>_audit.json` et `<Scene>_fr_missing.jsonl`. Vérification du gabarit :
`python docs/animations/render_series.py templates/scene_template.py TemplateScene --dry`
(données : `python docs/animations/templates/generate_template_data.py`).

## 7. Contrôle qualité et définition de « terminé »

**Images à regarder**, en anglais et en français, dans le mp4 1080p30 (ou un 480p15) : vers 1 s, à 35 %, à 65 %, juste avant
la carte de fin, et au milieu de la carte de fin. Extrayez-les avec PyAV (fourni avec Manim) ou n'importe quel lecteur.
Cherchez : logo qui ne recouvre rien, aucun texte qui se chevauche ou dépasse du cadre, aucun texte illisible à la taille
d'un écran d'ordinateur portable, tracés dessinés avant la phrase qui en parle.

**Json d'audit** (`<Scene>_<lang>_audit.json`) : `text_overlap`, `text_out_of_frame` et `logo_overlap` doivent être absents de
`warnings` ; `fr_autofit` (textes français réduits, jamais sous 0,8) doit être vide ou léger : raccourcissez la phrase plutôt
que de compter dessus ; `corner_overlaps` du coin choisi vide. Seul `endcard` (informatif) peut subsister.

**Liste de contrôle** (toutes les cases avant fusion) :

- [ ] Une idée, 2 temps au plus, 10 à 20 s de contenu à vitesse native ; le sous-titre donne le cadre.
- [ ] Chaque courbe et chaque nombre vient d'une vraie résolution bioptim stockée dans `data/<sujet>_*.npz` par
      `generate_<sujet>_data.py` ; statut IPOPT 0 pour toute solution montrée ; initialisation à chaud / continuation /
      minima locaux déclarés.
- [ ] Les entrées synthétiques sont étiquetées synthétiques à l'écran ; l'interface redessinée est étiquetée « redessiné, pas
      une capture d'écran ».
- [ ] Nombres à l'écran calculés à partir du npz dans la scène (pas de littéraux) ; les notes citent la sortie du générateur.
- [ ] Panneau de code : légende `Bioptim code` au-dessus, lignes vérifiées par `grep` dans cette version et identiques au
      générateur ; arguments qui changent colorés et animés.
- [ ] Une grandeur qui change a son fantôme de référence et son axe de différence ; rien n'est raconté qui ne soit tracé.
- [ ] Rôles des couleurs, polices, tailles, `CODE_X` / `CODE_W` / `TEXT_W` comme en section 3 ; unités entre parenthèses ;
      `t (s)`.
- [ ] Phrases entières, nombres dans la chaîne, pas de Paragraph, pas de LaTeX, pas d'abréviation sans développement,
      fondus simples.
- [ ] Bas de page de deux lignes au plus, finissant avant le logo ; `self.wait(2.5)` en dernier.
- [ ] Passe à sec OK (`--dry`) ; rendu EN : audit propre ; images vérifiées aux cinq instants.
- [ ] `i18n/fr_<sujet>.json` écrit ; `--lang fr --strict` sort avec 0 (0 clé manquante) ; glossaire respecté (commandes, états,
      nœud, tir multiple, collocation directe, bornes, coût...) ; décimales avec virgules ; images FR vérifiées, audit propre.
- [ ] Les liens se résolvent (`build_readme_tables.py --check`) et chaque plage de lignes a été ouverte et vérifiée.
- [ ] `notes/<sujet>.md` écrit avec réserves honnêtes et 2 à 3 exercices.
- [ ] Entrée de catalogue complète (deux langues, niveau, couple de sections, liens <= 5), tableaux des README régénérés et
      comptes en toutes lettres mis à jour.
- [ ] Formaté avec `black -t py311 -l120` ; `git status` ne montre que des fichiers texte de `docs/animations/` et de petits
      npz (ni mp4, ni png, ni `p.out`, ni dossier de sortie).

### Décisions (là où les conventions existantes étaient ambiguës)

1. **Helper du panneau de code** : les scènes utilisent soit `code_panel()` (features_scenes), soit un `code_block` local +
   légende de même aspect. Les nouvelles scènes utilisent `code_panel()` ; un panneau fait main n'est admis que pour une
   disposition qu'il ne sait pas exprimer, avec la légende au-dessus.
2. **Traduction de la légende** : README.md dit que la légende du panneau n'est « jamais traduite » ; en réalité `Bioptim code`
   a la clé `Code Bioptim` dans `i18n/fr.json`. Règle : la légende suit la traduction normale, les lignes de code jamais.
3. **Texte multiligne** : une phrase à retours à la ligne est un seul `Text` (`say`/`para`), une pile de lignes indépendantes
   est un `Text` par ligne (`Lines` dans la famille `anim_mhe.py`) ; `Paragraph` n'est pas utilisé dans les nouvelles scènes
   bien qu'`anim_markers.py` en garde un.
4. **`dec()`** : pour les étiquettes sans mot construites par la scène ; ni les phrases ni le code n'y passent.
5. **Objectif de durée** : 10 à 20 s de contenu / 2 temps pour les nouvelles vidéos ; les anciennes (jusqu'à 106 s au total)
   ne sont pas un modèle.
6. **Transition entre temps** : effacer tous les mobjects sauf le titre (`self.play(*[FadeOut(m) for m in self.mobjects if m
   is not title])`, puis `self.add(title)`), comme dans `ObjectivesNodes`.
7. **Gabarits hors des fichiers explorés** : `templates/` n'est exploré ni par `--gen-catalog` ni par `--all` ; une vraie scène
   est copiée en `docs/animations/anim_<sujet>.py`. Les données fictives du gabarit `data/template_demo.npz` sont ignorées
   par git ; le gabarit n'a pas de fichier de traduction française, donc `--lang fr --strict` y échoue par construction.
