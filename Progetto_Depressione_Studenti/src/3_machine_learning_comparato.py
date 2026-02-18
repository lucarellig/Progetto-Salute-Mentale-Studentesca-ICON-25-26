import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

print("="*60)
print(" FASE 3: ML COMPARATIVO, ANALISI ERRORI E SPIEGABILITÀ")
print("="*60)

# 1. Caricamento Dati
df = pd.read_csv("../data/dataset_arricchito.csv")
y = df['Depression']

# Separiamo le due versioni del dataset
X_base = df.drop(columns=['Depression', 'Onto_Rischio_Alto'])
X_onto = df.drop(columns=['Depression'])

# Salviamo i nomi delle feature per il grafico finale
feature_names_onto = X_onto.columns.tolist()

scaler = StandardScaler()
X_base_scaled = scaler.fit_transform(X_base)
X_onto_scaled = scaler.fit_transform(X_onto)

modelli = {
    "Albero Decisione": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Reg. Logistica": LogisticRegression(random_state=42)
}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# ==========================================
# FUNZIONE UNIFICATA PER VALUTAZIONE (Grezzi vs OntoBK)
# ==========================================
def valuta_e_crea_grafici(X, y, prefisso_titolo, prefisso_file):
    print(f"\n--- VALUTAZIONE: {prefisso_titolo.upper()} ---")
    print(f"{'Modello':<18} | {'Accuracy':<14} | {'Precision':<14} | {'Recall':<14} | {'F1-Score':<14}")
    print("-" * 85)
    
    risultati_metriche = []
    
    for nome, modello in modelli.items():
        res = cross_validate(modello, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        acc_m, acc_s = res['test_accuracy'].mean() * 100, res['test_accuracy'].std() * 100
        prec_m, prec_s = res['test_precision_macro'].mean() * 100, res['test_precision_macro'].std() * 100
        rec_m, rec_s = res['test_recall_macro'].mean() * 100, res['test_recall_macro'].std() * 100
        f1_m, f1_s = res['test_f1_macro'].mean() * 100, res['test_f1_macro'].std() * 100
        
        risultati_metriche.append({
            'Modello': nome, 'Accuracy': acc_m, 'Acc_std': acc_s,
            'Precision': prec_m, 'Prec_std': prec_s, 'Recall': rec_m, 'Rec_std': rec_s, 'F1-Score': f1_m, 'F1_std': f1_s
        })
        print(f"{nome:<18} | {acc_m:.2f} ±{acc_s:.2f}% | {prec_m:.2f} ±{prec_s:.2f}% | {rec_m:.2f} ±{rec_s:.2f}% | {f1_m:.2f} ±{f1_s:.2f}%")
        
    # Grafico a Barre Metriche
    df_metriche = pd.DataFrame(risultati_metriche)
    ax = df_metriche.set_index('Modello')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
        kind='bar', figsize=(12, 6), colormap='viridis', edgecolor='black'
    )
    plt.title(f"Metriche Avanzate - {prefisso_titolo}", fontweight='bold')
    plt.ylabel("Punteggio (%)")
    plt.ylim(0, 115) 
    plt.xticks(rotation=0)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4)
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    std_names = ['Acc_std', 'Prec_std', 'Rec_std', 'F1_std']
    for i, container in enumerate(ax.containers):
        labels = [f"{m:.1f} ±{s:.1f}%" for m, s in zip(df_metriche[metric_names[i]], df_metriche[std_names[i]])]
        ax.bar_label(container, labels=labels, padding=3, fontsize=8, fontweight='bold', rotation=90)

    plt.tight_layout()
    plt.savefig(f"{prefisso_file}_metriche.png")
    plt.close()

    # Learning Curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    for idx, (nome, modello) in enumerate(modelli.items()):
        train_sizes, train_scores, test_scores = learning_curve(modello, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')
        train_mean, train_std = np.mean(train_scores, axis=1) * 100, np.std(train_scores, axis=1) * 100
        test_mean, test_std = np.mean(test_scores, axis=1) * 100, np.std(test_scores, axis=1) * 100
        
        axes[idx].plot(train_sizes, train_mean, label="Training", color="darkorange", marker='o')
        axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="darkorange", alpha=0.2)
        axes[idx].plot(train_sizes, test_mean, label="Cross-Validation", color="navy", marker='o')
        axes[idx].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="navy", alpha=0.2)
        axes[idx].set_title(f"Learning Curve: {nome}", fontweight='bold')
        axes[idx].set_xlabel("Esempi")
        axes[idx].set_ylabel("Accuratezza (%)")
        axes[idx].legend(loc="lower right")
        axes[idx].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{prefisso_file}_learning_curves.png")
    plt.close()
    return risultati_metriche

# ==========================================
# ESECUZIONE 
# ==========================================
risultati_base = valuta_e_crea_grafici(X_base_scaled, y, "Dati Grezzi (No Ontologia)", "../docs/3/3a_base")
risultati_onto = valuta_e_crea_grafici(X_onto_scaled, y, "Dati con OntoBK", "../docs/3/3b_ontobk")

# ==========================================
# NUOVO: MATRICE DI CONFUSIONE (CROSS-VALIDATED)
# ==========================================
print("\nGenerazione Matrice di Confusione Cross-Validata (Random Forest + OntoBK)...")
rf_model = RandomForestClassifier(random_state=42)
# Predizioni su tutti i 10 fold
y_pred_cv = cross_val_predict(rf_model, X_onto_scaled, y, cv=cv)
cm = confusion_matrix(y, y_pred_cv)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sano (0)', 'Depresso (1)'], 
            yticklabels=['Sano (0)', 'Depresso (1)'])
plt.title("Matrice di Confusione (10-Fold CV)", fontweight='bold')
plt.xlabel("Diagnosi Predetta dal Modello")
plt.ylabel("Reale Stato Clinico")
plt.tight_layout()
plt.savefig("../docs/3/3c_matrice_confusione_cv.png")
plt.close()

# ==========================================
# NUOVO: FEATURE IMPORTANCE (SPIEGABILITÀ)
# ==========================================
print("Generazione Feature Importance (XAI)...")
# Addestriamo su tutto il dataset arricchito per estrarre i pesi
rf_model.fit(X_onto_scaled, y)
importances = rf_model.feature_importances_

# Creiamo un DataFrame, lo ordiniamo dal più importante al meno importante
df_imp = pd.DataFrame({'Feature': feature_names_onto, 'Importanza': importances})
df_imp = df_imp.sort_values(by='Importanza', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importanza', y='Feature', data=df_imp, palette='viridis')
plt.title("Importanza delle Variabili (Random Forest + OntoBK)", fontweight='bold')
plt.xlabel("Peso sul processo decisionale")
plt.ylabel("Variabile")
plt.tight_layout()
plt.savefig("../docs/3/3d_feature_importance.png")
plt.close()

# ==========================================
# NUOVO: CURVA PRECISION-RECALL (CROSS-VALIDATED)
# ==========================================
print("Generazione Curva Precision-Recall (Random Forest + OntoBK)...")
# Otteniamo le probabilità predette in cross-validation (colonna 1 per la classe "Depresso")
y_prob_cv = cross_val_predict(rf_model, X_onto_scaled, y, cv=cv, method='predict_proba')[:, 1]

# Calcoliamo Precision, Recall e l'Average Precision Score (AP)
precision, recall, _ = precision_recall_curve(y, y_prob_cv)
ap_score = average_precision_score(y, y_prob_cv)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='purple', lw=2, label=f'Random Forest (AP = {ap_score:.2f})')
plt.xlabel('Recall (Tasso di veri positivi trovati)')
plt.ylabel('Precision (Tasso di predizioni corrette)')
plt.title('Curva Precision-Recall (10-Fold CV)', fontweight='bold')
plt.legend(loc='lower left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("../docs/3/3e_curva_precision_recall.png")
plt.close()

print("\nFase 3 completata, grafici clinici salvati ('3c_matrice_cv', '3d_feature_importance', '3e_curva_precision_recall').")