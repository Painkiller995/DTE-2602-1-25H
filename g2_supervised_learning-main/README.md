<!-- # Karaktersatt oppgave 2 | DTE-2602  -->
# DTE-2602 Karaktersatt oppgave 2: Veiledet læring

## Introduksjon
Hensikten med denne karaktersatte oppgaven er at dere skal implementere og bruke
algoritmer for veiledet læring ("supervised learning"). Besvarelsen bør både
demonstrere _at_ slik veiledet læring er mulig, men også _hvordan_ veiledet læring
fungerer _under ulike omstendigheter_, ved bruk av ulike algoritmer eller ulike
parametere. 

Som et eksempel på et datasett skal vi bruke "Palmer penguins" som dere antakelig
allerede kjenner fra før fra innlevering 6. Datasettet beskriver diverse egenskaper ved
et sett med pingviner, og hvilken art pingvinene tilhører. Dere skal bruke veiledet
læring for å artsbestemme pingvinene ("klassifisering"). Merk at datasettet bare er et
nyttig eksempel på data fra den virkelige verden - et middel for å demonstrere mer generelle
teknikker. Du finner datasettet i fila ``palmer_penguins.csv`` i repository for oppgaven.

<img
src="https://github.com/allisonhorst/palmerpenguins/blob/main/man/figures/lter_penguins.png?raw=true"
alt="Drawing" style="width: 600px;"/>

Figuren under viser ulike varianter av "scatterplott" for de fire numeriske størrelsene
i datasettet:
- bill_length_mm
- bill_depth_mm
- flipper_length_mm
- body_mass_g

Plottene viser alle parvise kombinasjoner av disse fire. Diagonalen viser histogram for
enkelt-størrelser. Legg merke til hvordan noen kombinasjoner gir punktskyer der de tre
klassene (pingvinartene) overlapper mye, mens andre kombinasjoner gir bedre separasjon.

![Scatter-plott av Palmer penguins](https://seaborn.pydata.org/_images/pairplot_3_0.png)
_Figur hentet fra
https://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn.pairplot_




## Oppgaver
I denne oppgaven skal dere selv implementere diverse funksjoner for å lese inn og
preprosessere data, samt to klasser for supervised learning: "Perceptron" og
"DecisionTree". Dere skal også sette opp og vurdere deres egne eksperimenter ved å bruke
disse klassene. *Obs: Ikke skriv rapporten slik at dere svarer på oppgavene punkt for
punkt - beskriv helheten.* 

Startkode for implementasjonen er oppgitt i filene `supervised_learning.py` (fylles ut
med egen kode) og
`decision_tree_nodes.py` (trenger ikke endres). 

### Forberede data og vurdere resultater
- Les inn datasettet fra `palmer_penguins.csv`. Fjern alle rader som mangler data, bruk
  de fire kolonnene med numeriske data, og normaliser disse med Z-score, som oppgitt i
  startkoden.  Implementeres som funksjon `read_data()`. 
- Implementer en funksjon `convert_y_to_binary()` for å konvertere et datasett med 3
  eller flere klasser til et binært datasett. 
- Implementer en funksjon `train_test_split()` for å splitte et datasett ($X$,$y$) opp i
  et tranings-datasett ($X_{\text{train}}$, $y_{\text{train}}$) og et test-datasett
  ($X_{\text{test}}$, $y_{\text{test}}$).  
- Lag en funksjon `accuracy()` som tar en vektor $y_{\text{true}}$ med "sanne" verdier
  og en vektor $y_{\text{pred}}$ med estimerte verdier som input. Funksjonen skal
  beregne nøyaktigheten til modellen som produserte $y_{\text{pred}}$.


### Perceptron
- Implementer en klasse `Perceptron`. Klassen skal ha metodene `.train()`,
  `.predict_single()` og `.predict()`, som indikert i startkoden. Skriv docstrings for
  alle metodene.


### Decision tree
- Implementer funksjonene ``gini_impurity()``, ``gini_impurity_reduction()`` og
  ``best_split_feature_value()`` som vist i startkoden. Merk: Noen enkle tester for å
  disse funksjonene er oppgitt i fila `decision_tree_test.py`.
- Implementer en klasse ``DecisionTree`` som bruker metodene knyttet til Gini impurity.
  Deler av implementasjonen er allerede ferdig og er oppgitt i startkoden. Du skal
  implementere metoden ``._predict()``, som bruker et trent beslutningstre for å
  klassifisere nye data. 

 
### Eksperimenter
#### Perceptron - 1

- Les inn Palmer Penguins-datasettet med `read_data()`, og bruk `train_test_split()` for
 å splitte det opp i et treningsdatasett
  ($X_{\text{train}}$, $y_{\text{train}}$) og et testdatasett ($X_{\text{test}}$,
  $y_{\text{test}}$). Gi en begrunnelse for hvor stor andel av datapunktene du legger i
  hver av delene. 
- Bruk datasettet ($X_{\text{train}}$, $y_{\text{train}}$) til å trene en
  Perceptron-modell til å skille pingvinarten **Gentoo** fra de to andre artene. Du skal
  kun bruke de to kolonnene _bill_depth_mm_ og _flipper_length_mm_ fra X-matrisa.  Du kan
  bruke funksjonen `convert_y_to_binary()` for å konvertere pingvinartene til to klasser. Bruk
  datasettet ($X_{\text{test}}$, $y_{\text{test}}$) sammen med funksjonen `accuracy()`
  for å måle nøyaktigheten til modellen. 
- Visualiser "decision boundary" for modellen i oppgave over. Lag et plott som viser
  hver pingvin i $X_{\text{train}}$ som et punkt (bruk `matplotlib.pyplot.scatter()`). Plott så decision
  boundary for modellen som en rett linje (bruk hjelpemetoden
  `decision_boundary_slope_intercept()`, og `matplotlib.pyplot.plot()`). Linja bør
  ligge mellom Gentoo-punktene og punktene for de to andre artene. 

#### Perceptron  - 2
- Lag et nytt perceptron som skal skille arten **Chinstrap** fra de to andre. Bruk kun
  kolonnene _bill_length_mm_ og _bill_depth_mm_ fra X-matrisa (ikke samme som over!).
  Tren perceptron'et med ($X_{\text{train}}$, $y_{\text{train}}$).  Merk at det ikke er
  sikkert at modellen konvergerer - forklar i så fall hvorfor. Visualiser "decision
  boundary" på samme måte som over. Mål nøyaktigheten til modellen med
  ($X_{\text{test}}$, $y_{\text{test}}$). 

#### Decision tree - 1
- Lag et decision tree for å skille pingvinarten **Gentoo** fra de to andre artene
  basert på _bill_depth_mm_ og _flipper_length_mm_, på samme måte som med perceptron.
  Mål nøyaktigheten og visualisér treet (rot, greinnoder, løvnoder). Kommentér: Gir
  verdiene til grein- og løvnodene mening?

#### Decision tree - 2
- Lag et decision tree for å skille pingvinarten **Chinstrap** fra de to andre artene
  basert på _bill_length_mm_ og _bill_depth_mm_, på samme måte som med perceptron. Mål
  nøyaktigheten og visualisér treet. Sammenlign med resultatene for perceptron. 

#### Decision tree - 3
- Gjenta følgende eksperiment flere ganger, med en ny tilfeldig "stokking" og oppdeling
  av ($X_{\text{train}}$, $y_{\text{train}}$) og ($X_{\text{test}}$, $y_{\text{test}}$)
  for hver gang: Lag et decision tree basert på alle 4 features i datasettet, som skal
  skille mellom alle 3 arter i datasettet. Bruk ($X_{\text{test}}$, $y_{\text{test}}$)
  for å måle nøyaktigheten. Oppgi statistikk for resultatene.
- Kommentér i rapporten: Hvorfor er klassifisering av mer enn to kategorier mulig med et
  decision tree, og ikke med et perceptron?

#### Sammenligning og oppsummering
- Sammenlign resultatene for de ulike modellene du har trent og testet. Er det
  forskjeller mellom resultatene? Hva er sannsynlige grunner til det, i så fall? 



## Levering og vurdering av oppgaven
Besvarelsen på oppgaven består av to deler som skal leveres på Canvas.
- Python-fil `supervised_learning.py` med implementasjon. Implementasjonen kan bruke biblioteker
  som er inkludert i Python 3.10, samt ``binarytree``, ``numpy``, ``matplotlib`` og
  ``scipy.stats``. _Ingen andre eksterne biblioteker skal brukes._ 
- En rapport i form av et PDF-dokument (f.eks. generert fra MS Word eller LaTeX) med
  beskrivelse av din egen løsning. Rapporten skal være mellom 4 og 10 sider lang
  (forside ikke inkludert). Rapporten kan skrives på norsk eller engelsk. 

Vurderingen gjøres ved bruk av vurderingsveiledning ("rubrikk") på Canvas, der poeng
tildeles for ulike kriterier knyttet til rapporten og koden.