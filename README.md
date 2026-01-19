HeartGuard

# Progetto AI – Predizione degli Attacchi Cardiaci

Progetto di **Intelligenza Artificiale**  
Anno accademico **2025/2026**  
Corso di Laurea Magistrale in **Sicurezza Informatica e Tecnologie Cloud (LM-66)**

**Docenti:**
- Prof.ssa Loredana Caruccio  
- Prof.ssa Genoveffa Tortora  

---

## 📌 Descrizione del progetto

Negli ultimi anni si è registrato un significativo aumento dei decessi causati da **attacchi cardiaci**, uno degli eventi cardiovascolari più gravi e potenzialmente letali. Una **diagnosi preventiva** può risultare fondamentale per la sopravvivenza dei pazienti; tuttavia, l’identificazione precoce di un attacco cardiaco è un compito complesso, poiché richiede l’analisi di molteplici fattori clinici e i sintomi possono variare notevolmente da persona a persona, manifestandosi anche in forme atipiche.

Con l’introduzione del **Machine Learning**, è possibile valutare l’efficacia di diversi modelli di apprendimento automatico nel **prevedere il rischio di attacco cardiaco**, classificando i pazienti in due categorie:
- **Attacco cardiaco**
- **Non attacco cardiaco**

Per la valutazione delle prestazioni dei modelli vengono utilizzate metriche quali:
- Accuracy
- Precision
- Recall
- F1-Score  

oltre a strumenti di analisi avanzata come:
- **Matrice di confusione**
- **Curva AUC-ROC**

L’obiettivo principale del progetto è **individuare il miglior modello predittivo**, confrontando i risultati ottenuti dai diversi algoritmi e analizzandone **punti di forza e limitazioni**.

---

## 📁 Struttura del progetto

```
Progetto_AI-main/
│
├── Deliverables/
│   └── progetto/
│       ├── main.py
│       └── dataset/
│           └── Medicaldataset.csv
│
├── File/
│   └── Progetto_AI.pdf
│
└── README.md
```

---

## 📊 Dataset

Il dataset utilizzato (`Medicaldataset.csv`) contiene dati di natura medica ed è impiegato per un problema di **classificazione binaria**.

Le principali operazioni di preprocessing includono:
- caricamento dei dati con **Pandas**
- suddivisione in **training set** e **test set**
- normalizzazione delle feature tramite **StandardScaler**

---

## 🤖 Modelli di Machine Learning

- Logistic Regression  
- Decision Tree Classifier  

Gli iperparametri vengono ottimizzati tramite `RandomizedSearchCV`.

---

## 📈 Valutazione

I modelli vengono valutati su training e test set utilizzando:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve e AUC

---

## ▶️ Esecuzione

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python main.py
```

---

## 📄 Documentazione

La relazione completa è disponibile nel file **Progetto_AI.pdf**.

---

## 👤 Autore

**Armando Imbimbo**  
Laurea Magistrale in Sicurezza Informatica e Tecnologie Cloud (LM-66)

---

## 📌 Note

Progetto a scopo didattico e sperimentale.
