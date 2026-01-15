import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler


# ------------------ Ignora tutti i warning ------------------ #
warnings.filterwarnings("ignore")

# ------------------ Funzioni ------------------ #
def dataset_analysis(df):
    df.info()
    missing_values = df.isnull().any()
    print("🔍 Missing Values Check:")
    print(missing_values if missing_values.sum() > 0 else "No missing values found! ✅")
    duplicate = df.duplicated().sum()
    print("🔍 Duplicate Check:")
    print(duplicate if duplicate.sum() > 0 else "No duplicates found! ✅")

def dataset_distribution(df):
    print("\n--- ANALISI Heart Attack ---")
    total_rows = len(df)
    counts = df['Result'].value_counts()
    print("\nConteggio Heart Attack:")
    print(counts)
    percentages = (counts / total_rows) * 100
    print("\nPercentuali Heart Attack:")
    print(percentages.round(2).astype(str) + "%")
    labels = ['No Heart Attack (0)', 'Heart Attack (1)']
    sizes = [counts[0], counts[1]]
    plt.figure(figsize=(10, 10))
    plt.pie(
        sizes,
        labels=labels,
        autopct=lambda p: f'{p:.2f}%',
        startangle=90
    )
    plt.title("Distribuzione percentuale Heart Attack")
    plt.axis('equal')
    plt.show()



def show_graphics(df):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, col in enumerate(df.columns):
        row, col_position = divmod(i, 3)
        sns.boxplot(data=df, y=col, ax=axes[row, col_position])
        plt.tight_layout()
    plt.show()


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def print_metrics(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Heart Attack', 'Heart Attack'])
    disp.plot(cmap="Blues")
    plt.subplots_adjust(left=0.25)
    plt.title(f"Matrice di Confusione - {dataset_name}")
    plt.show()
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n--- Metriche {dataset_name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")


def matrice_di_cor(df):
    #correlation matrix
    correlation_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    #creare le heatmap con Seaborns
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='RdBu_r',
        vmin=-1, vmax=1,
        cbar_kws={'label': 'Correlation'},
        fmt='.2f',
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Correlation Heatmap of Features', fontsize=18) #Titolo
    plt.tight_layout()
    plt.show()

def the_best_feature(df):
    #identificare le features maggiormente correlate con il taget
    correlations = df.corr(numeric_only=True)['Result'].drop('Result').abs().sort_values(ascending=True)

    #Stampa le migliori features
    print("Features ranked by absolute correlation with Heart Attack:")
    for feature, corr in correlations.items():
        print(f"{feature}: {corr:.4f}")

    #Plot
    plt.figure(figsize=(10, 8))
    plt.barh(correlations.index,correlations.values,color=plt.cm.viridis(correlations.values / correlations.values.max()))
    plt.xlim(0, correlations.values.max() * 1.25) #aggiungi spazio a destra

    #aggiunta valori alle bars
    for index, value in enumerate(correlations.values):
        plt.text(value + 0.01, index, f"{value:.4f}", va='center', fontsize=12)

    # Labels e title
    plt.xlabel('Absolute Correlation Value')
    plt.ylabel('Features')
    plt.title('The Features Most Correlated with Heart Attack', fontsize=16)

    # Display plot
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, model_name, y_score=None):
    plt.figure(figsize=(7, 6))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    print(f"AUC ROC for {model_name}: {roc_auc:.4f}")


# ------------------ Caricamento, analisi del dataset, mapping e rimozione outlier ------------------ #
ds = pd.read_csv('C:/Users/armi9/PycharmProjects/progetto/dataset/Medicaldataset.csv')

# Analisi iniziale
dataset_analysis(ds)
d = {'positive': 1, 'negative': 0} # Mappatura target
ds['Result'] = ds['Result'].map(d)
dataset_distribution(ds) #distribuzione del dataset
show_graphics(ds) #outliers

#prima della rimozione degli outlier
matrice_di_cor(ds)
the_best_feature(ds)

# Rimozione outlier
for col in ds.columns:
    ds = remove_outliers(ds, col)

#dopo la rimozione degli outlier
matrice_di_cor(ds)
the_best_feature(ds)


# ------------------ Divisione train/test + Feature scaling ------------------ #
X = ds.drop('Result', axis=1)
y = ds['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

#selezione delle feature numeriche
numeric_features = ['Age', 'Gender', 'Heart rate', 'Systolic blood pressure',
					'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']
scaler = StandardScaler() #inizailizza scaler

scaler.fit(X_train[numeric_features])

X_train_scaled = scaler.transform(X_train[numeric_features])
X_test_scaled = scaler.transform(X_test[numeric_features])

# ------------------ Logistic Regression ------------------ #
logreg_unscaled = LogisticRegression(max_iter=1000, class_weight='balanced')
param_distributions = {'penalty': ['l1', 'l2'],
                       'solver': ['saga', 'liblinear'],
                       'C': list(np.arange(1, 21))}

random_search_lr = RandomizedSearchCV(
    estimator=logreg_unscaled,
    param_distributions=param_distributions,
    scoring='accuracy',
    cv=10,
    random_state=42
)


random_search_lr.fit(X_train, y_train)
best_logreg_lr = random_search_lr.best_estimator_

# Predizioni train e test senza StandardScaler
y_train_pred_log = best_logreg_lr.predict(X_train)
y_test_pred_log = best_logreg_lr.predict(X_test)

print("\n=== Logistic Regression senza StandardScaler ===")
print("Migliori parametri trovati:", random_search_lr.best_params_)
print_metrics(y_train, y_train_pred_log, "Train")
print_metrics(y_test, y_test_pred_log, "Test")
plot_roc_curve(y_test, "Logistic Regression (UnScaled)", y_score=best_logreg_lr.predict_proba(X_test)[:,1])

# ------------------ Logistic Regression con StandardScaler------------------ #
logreg_scaled = LogisticRegression(max_iter=1000, class_weight='balanced')

random_search_lr_scaled = RandomizedSearchCV(
    estimator=logreg_scaled,
    param_distributions=param_distributions,
    scoring='accuracy',
    cv=10,
    random_state=42
)

random_search_lr_scaled.fit(X_train_scaled, y_train)
best_logreg_lr_scaled = random_search_lr_scaled.best_estimator_

# Predizioni train e test con scaled
y_train_pred_log_scaled = best_logreg_lr_scaled.predict(X_train_scaled)
y_test_pred_log_scaled = best_logreg_lr_scaled.predict(X_test_scaled)

print("\n=== Logistic Regression ===")
print("Migliori parametri trovati:", random_search_lr_scaled.best_params_)
print_metrics(y_train, y_train_pred_log_scaled, "Train")
print_metrics(y_test, y_test_pred_log_scaled, "Test")
plot_roc_curve(y_test, "Logistic Regression (Scaled)", y_score=best_logreg_lr_scaled.predict_proba(X_test_scaled)[:,1])


# ------------------ Decision Tree ------------------ #

dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# Parametri
param_dist = {
    'max_depth': np.arange(3, 16),          # profondità 3–15
    'min_samples_split': np.arange(2, 21),  # split minimo 2–20
    'min_samples_leaf': np.arange(1, 12),   # foglie 1–11
    'max_features': ['sqrt'],               # meno feature per split
    'criterion': ['gini', 'entropy']        # aggiunta del criterio
}

# RandomizedSearchCV ottimizzando
random_search_dt = RandomizedSearchCV(
    estimator=dt,
    param_distributions=param_dist,
    n_iter=20,              # numero di combinazioni da testare
    scoring='accuracy',     # massimizza accuracy
    cv=10,                   # cross-validation standard
    random_state=42,
)


#random_search_dt.fit(X_train_scaled, y_train)
random_search_dt.fit(X_train, y_train) #addestra il modello
best_dt=random_search_dt.best_estimator_


# Predizioni
#y_train_pred_dt = best_dt.predict(X_train_scaled) #non ha necessità di essere scalato
#y_test_pred_dt = best_dt.predict(X_test_scaled)
y_train_pred_dt = best_dt.predict(X_train)
y_test_pred_dt = best_dt.predict(X_test)

print("\n=== Decision Tree ===")
print("Best parameters found:", random_search_dt.best_params_)
print_metrics(y_train, y_train_pred_dt, "Train")
print_metrics(y_test, y_test_pred_dt, "Test")
plot_roc_curve(y_test, "Decision Tree", y_score=best_dt.predict_proba(X_test)[:,1])

