# Riconoscimento di testi italiani generati da AI

Progetto per il corso di **Linguistica Computazionale II** (Informatica Umanistica, Università di Pisa, a.a. 2025/2026).

Il task è la **classificazione binaria di testi in italiano**: distinguere i testi scritti da esseri umani (classe 0) da quelli generati da modelli di linguaggio (classe 1). I dati provengono dalla shared task **DeSegMa-it** (subTask A). Il problema viene affrontato con quattro approcci di complessità crescente, dal machine learning classico con feature ingegnerizzate fino al fine-tuning di un transformer.

## Risultati

Accuracy e F1 macro sul test set (1000 documenti bilanciati):

| # | Approccio | Accuracy (val) | Accuracy (test) | F1 macro (test) |
|---|-----------|:---:|:---:|:---:|
| 1 | SVM lineare + n-grammi (migliore config.: unigrammi di parola) | 0.97 | 0.88 | 0.88 |
| 2 | SVM lineare + feature linguistiche non lessicali (Profiling-UD) | 0.94 | 0.88 | 0.88 |
| 3 | SVM lineare + word embeddings (itWaC) | 0.93 | 0.80 | 0.79 |
| 4 | Fine-tuning di BERT (`dbmdz/bert-base-italian-cased`, 3 epoche) | 0.99 | **0.95** | **0.95** |

Il calo tra validation e test per gli approcci 1–3 suggerisce uno shift di distribuzione tra le due partizioni del dataset; BERT è il modello che generalizza meglio. L'analisi completa (configurazioni sperimentate, cross validation, feature più informative, curve di loss) è nel [report](Report_LCII.pdf).

## Struttura del repository

| File | Contenuto |
|------|-----------|
| [`0_creazione_corpus.ipynb`](0_creazione_corpus.ipynb) | Costruzione di training (2000 doc.), validation e test set (1000 doc. ciascuno, bilanciati) dai dati della shared task |
| [`1_svm_ngrams.ipynb`](1_svm_ngrams.ipynb) | SVM lineari su n-grammi di parola, lemma, PoS (1–3) e carattere (3–5), selezione del modello con 5-fold CV, analisi dei coefficienti; test extra su testo generato da ChatGPT |
| [`2_svm_feature_linguistiche.ipynb`](2_svm_feature_linguistiche.ipynb) | SVM lineari su ~130 feature linguistiche non lessicali estratte con [Profiling-UD](http://linguistic-profiling.italianlp.it/) (lunghezze, ricchezza lessicale, distribuzioni PoS, sintassi) |
| [`3_svm_word_embeddings.ipynb`](3_svm_word_embeddings.ipynb) | SVM lineari su rappresentazioni ottenute aggregando word embeddings itWaC (tre strategie di aggregazione confrontate) |
| [`4_finetuning_bert.ipynb`](4_finetuning_bert.ipynb) | Fine-tuning di BERT italiano per 3 epoche con PyTorch, valutazione per epoca e curve di loss |
| [`Classes.py`](Classes.py) | Utility condivise: classi `Token`/`Document`, annotazione con Stanza, estrazione e filtraggio di n-grammi, normalizzazione del testo, caricamento embeddings |
| `TRAINUD.csv`, `VALUD.csv`, `TESTUD.csv` | Feature linguistiche estratte con Profiling-UD per le tre partizioni (solo valori numerici aggregati, nessun testo) |
| [`Report_LCII.pdf`](Report_LCII.pdf) | Report del progetto con metodologia e discussione dei risultati |

## Riproducibilità

I notebook sono salvati **con gli output**, quindi i risultati sono consultabili senza eseguire nulla.

Per rieseguire il progetto servono alcune risorse non incluse nel repository:

1. **Dataset DeSegMa-it (subTask A)** — non ridistribuibile, va richiesto agli organizzatori della shared task. I file attesi da `0_creazione_corpus.ipynb` sono `trainset/desegma-it.subTaskA.shared.train.*.csv` e `testset/desegma-it.subTaskA.with_labels.test.*.csv`; il notebook produce `df_train.csv`, `df_val.csv`, `df_test.csv` usati dagli altri notebook.
2. **Word embeddings itWaC** (`embeddits.sqlite`) — disponibili tra le [risorse di ItaliaNLP Lab](http://www.italianlp.it/resources/italian-word-embeddings/), usati dal notebook 3.
3. I file `*_documents.pkl` (documenti annotati con Stanza) vengono rigenerati dal notebook 1.

Installazione delle dipendenze:

```bash
pip install -r requirements.txt
```
