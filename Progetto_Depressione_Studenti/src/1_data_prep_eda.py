import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype
import warnings

# Ignoriamo i warning per tenere pulito il terminale
warnings.filterwarnings("ignore")

print("FASE 1: Data Engineering e Analisi Esplorativa (EDA)...")

# 1. CARICAMENTO DATI
df = pd.read_csv("../data/Student Depression Dataset.csv") 

# 2. PULIZIA DATI
if 'id' in df.columns:
    df = df.drop(columns=['id'])

print("Imputazione dei valori mancanti...")
for col in df.columns:
    if is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Encoding delle variabili testuali...")
le = LabelEncoder()
for col in df.columns:
    if not is_numeric_dtype(df[col]):
        df[col] = df[col].astype(str) 
        df[col] = le.fit_transform(df[col])

# Salvataggio dataset pulito
df.to_csv("../data/dataset_pulito.csv", index=False)
print("Dataset pulito e salvato come 'dataset_pulito.csv'.")

# 3. GENERAZIONE GRAFICI EDA
print("Generazione dei grafici esplorativi in corso...")
sns.set_theme(style="whitegrid")

# Grafico 1: Torta
plt.figure(figsize=(6, 6))
df['Depression'].value_counts().plot.pie(labels=['Non Depresso (0)', 'Depresso (1)'], autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
plt.title("Distribuzione Depressione", fontweight='bold')
plt.savefig("../docs/1/1a_eda_torta.png")
plt.close()

# Grafico 2: Matrice di Correlazione (CON NUMERI LEGIBILI)
plt.figure(figsize=(14, 12)) # Aumentiamo lo spazio per far respirare i numeri
# annot=True accende i numeri, fmt='.2f' taglia a 2 decimali, annot_kws riduce la grandezza del testo
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap="coolwarm", annot_kws={"size": 8})
plt.title("Matrice di Correlazione", fontweight='bold')
plt.tight_layout()
plt.savefig("../docs/1/1b_eda_correlazione.png")
plt.close()

# Convertiamo le colonne in interi per rimuovere i decimali (.0) e far combaciare l'ordine
df['Financial Stress'] = df['Financial Stress'].astype(int)
df['Academic Pressure'] = df['Academic Pressure'].astype(int)

# Grafico 3: Stress Finanziario vs Depressione
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Financial Stress', hue='Depression', palette=['#66b3ff', '#ff9999'], order=[1, 2, 3, 4, 5])
plt.title("Impatto dello Stress Finanziario sulla Depressione", fontweight='bold')
plt.xlabel("Livello di Stress Finanziario (1 = Basso, 5 = Alto)")
plt.ylabel("Numero di Studenti")
plt.legend(title="Depressione", labels=["No", "Sì"])
plt.savefig("../docs/1/1c_eda_stress_finanziario.png")
plt.close()

# Grafico 4: Pressione Accademica vs Depressione
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Academic Pressure', hue='Depression', palette=['#66b3ff', '#ff9999'], order=[1, 2, 3, 4, 5])
plt.title("Impatto della Pressione Accademica sulla Depressione", fontweight='bold')
plt.xlabel("Livello di Pressione Accademica (1 = Basso, 5 = Alto)")
plt.ylabel("Numero di Studenti")
plt.legend(title="Depressione", labels=["No", "Sì"])
plt.savefig("../docs/1/1d_eda_pressione_accademica.png")
plt.close()

print("\nFASE 1 COMPLETATA! Grafici corretti e salvati.")