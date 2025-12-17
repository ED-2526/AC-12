
**RECOMMENDER SYSTEMS FOR ONLINE SHOPPING (AMAZON)**


*Descripció del Projecte*

L’objectiu d’aquest treball és predir les recomanacions de productes d'Amazon mitjançant tècniques d’aprenentatge automàtic. Utilitzant un conjunt de dades (dataaset) que inclou informació sobre els usuaris, els productes, les valoracions i el moment temporal de les interaccions (timestamp). Pretenem identificar patrons de comportament de compra i preferències dels usuaris dins la plataforma.

L’anàlisi es basa en models de regressió i filtratge col·laboratiu per predir les valoracions dels productes, així com en tècniques de clustering per identificar grups d’usuaris amb característiques i interessos similars, amb l’objectiu de millorar la precisió i la personalització de les recomanacions.


*Objectius del Projecte*
1. Predir les valoracions dels productes d’Amazon mitjançant models de recomanació col·laborativa, utilitzant tècniques d’aprenentatge automàtic com KNN Item-Item i FunkSVD (amb i sense bias).
2. Analitzar si els patrons de comportament dels usuaris i les característiques dels productes generen estructures o agrupacions implícites, observant si els factors latents apresos pels models permeten identificar similituds entre usuaris i ítems, i si existeix una segmentació clara basada en les preferències de compra.
3. Desenvolupar perfils predictius d’usuaris i productes a partir de les valoracions històriques, els factors latents i les relacions de similitud, amb l’objectiu de caracteritzar els interessos dels usuaris i la popularitat o especialització dels productes.
4. Avaluar el rendiment dels diferents models de recomanació mitjançant mètriques estàndard com Precision@K, Recall@K, MAP@K, NDCG@K i RMSE, comparant els resultats obtinguts amb diferents estratègies d’avaluació (Leave-One-Out, Cross-Validation i biblioteques externes, com lenskit i surprise).


*Contingut del Repositori*

  - data_cleaner.py — Script per carregar, netejar i filtrar el dataset.
  - train_knn.py — Entrenament del model Item-Item KNN amb similitud cosinus.
  - train_svd.py — Entrenament de FunkSVD bàsic (sense bias) mitjançant SGD.
  - train_svd_bias.py — Entrenament de FunkSVD amb bias (global, usuari i ítem).
  - infer_knn.py — Predicció de ratings i generació de recomanacions top-N amb el model KNN.
  - infer_svd.py — Predicció i recomanacions amb models SVD (detecta automàticament si inclou bias).
  - main_leaveoneout.py — Avaluació amb Leave-One-Out, càlcul de mètriques i gràfics de comparació.
  - main_crossval.py — Avaluació amb K-Fold Cross-Validation per usuari.
  - main_surprise.py — Validació externa utilitzant la biblioteca Surprise (KNNBasic i SVD).
  - lenskit.py
  - ROCcurve.py
  - cleaned_data.csv — Dataset netejat generat (no inclòs al repo per mida, generar amb data_cleaner.py).


*Descripció del Dataset*

Font: Dataset públic d'Amazon Reviews (categoria Electronics), provinent de Kaggle (https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews).
Nom: ratings_Electronics(1).csv
Nombre de registres: Aproximadament 7.824.482 valoracions.
Columnes (4):
- userID: Identificador de l'usuari (string).
- itemID: Identificador del producte (ASIN, string).
- rating: Valoració (1 a 5, float).
- timestamp: Data de la valoració (Unix timestamp).

*Requisits i Instal·lació*

Python 3.8+
Llibreries necessàries:
- pandas
- numpy
- pickle
- os
- scikit-learn
- matplotlib
- surprise  

*Autors*

Clàudia Blasco, Laura Buide i Lucía Rodríguez.
