
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
  - data_cleaner.py
  - train_knn.py
  - train_svd.py
  - train_svd_bias.py
  - infer_knn.py
  - infer_svd.py
  - main_surprise.py
  - main_leaveoneout.py
  - main_crossval.py
  - lenskit.py
  - ROCcurve.py


*Descripció del Dataset*
Nom: ratings_Electronics(1).csv
Nombre de registres: 1.000.000
Nombre de columnes: 4
