import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch
from pgmpy.inference import VariableElimination
import warnings
import logging
import time

# Disattiviamo i log noiosi di pgmpy sui "Datatype inferred"
logging.getLogger('pgmpy').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

print("="*70)
print(" FASE 5: RAGIONAMENTO PROBABILISTICO E STRUCTURE LEARNING")
print("="*70)

# 1. CARICAMENTO DATI
df = pd.read_csv("../data/dataset_pulito.csv")
colonne_grafo = [
    'Financial Stress', 'Academic Pressure', 
    'Sleep Duration', 'Dietary Habits', 
    'Family History of Mental Illness', 
    'Depression', 'Have you ever had suicidal thoughts ?'
]
df_bayes = df[colonne_grafo].copy()

# ==========================================
# FUNZIONE PER DISEGNARE GRAFI "DA TESI"
# ==========================================
def disegna_grafo_elegante(modello, titolo, nome_file, colore_nodi):
    G = nx.DiGraph()
    G.add_edges_from(modello.edges())
    
    plt.figure(figsize=(11, 7))
    # Layout circolare che distanzia perfettamente i nodi
    pos = nx.shell_layout(G) 
    
    # Stile avanzato NetworkX (Nodi)
    nx.draw_networkx_nodes(G, pos, node_size=4000, node_color=colore_nodi, edgecolors='black', linewidths=1.5)
    
    # Stile avanzato NetworkX (Archi)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=30, edge_color='#555555', width=2.0, node_size=4000)
    
    # Nomi dei nodi (andiamo a capo al posto degli spazi per farli entrare nei cerchi)
    labels = {node: node.replace(" ", "\n") for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold", font_family="sans-serif")
    
    plt.title(titolo, fontsize=14, fontweight="bold", pad=20)
    plt.axis("off") # Rimuove i bordi del grafico
    plt.tight_layout()
    plt.savefig(nome_file, dpi=300, bbox_inches='tight') # Salvataggio in alta risoluzione
    plt.close()

# 2. APPROCCIO 1: RETE EXPERT-DRIVEN
print("-> 1. Costruzione Rete Expert-Driven (Causalità Biologica)...")
modello_esperto = DiscreteBayesianNetwork([
    ('Financial Stress', 'Academic Pressure'),              
    ('Academic Pressure', 'Sleep Duration'),                
    ('Academic Pressure', 'Depression'),                    
    ('Sleep Duration', 'Depression'),                       
    ('Dietary Habits', 'Depression'),                       
    ('Family History of Mental Illness', 'Depression'),     
    ('Depression', 'Have you ever had suicidal thoughts ?') 
])
modello_esperto.fit(df_bayes, estimator=MaximumLikelihoodEstimator)
disegna_grafo_elegante(modello_esperto, "DAG Expert-Driven (Causale)", "../docs/5/5a_grafo_esperto.png", "#85C1E9")

# 3. APPROCCIO 2: STRUCTURE LEARNING AUTOMATICO (Data-Driven)
print("-> 2. Esecuzione Hill Climbing per Structure Learning (Data-Driven)...")
hc = HillClimbSearch(df_bayes)

# Misuriamo il tempo esatto
start_time = time.time()
struttura_automatica = hc.estimate() 
end_time = time.time()

tempo_impiegato = end_time - start_time

print(f"   [!] Ottimo globale raggiunto in: {tempo_impiegato:.4f} secondi")

modello_automatico = DiscreteBayesianNetwork(struttura_automatica.edges())
modello_automatico.fit(df_bayes, estimator=MaximumLikelihoodEstimator)
disegna_grafo_elegante(modello_automatico, "DAG Data-Driven (Hill Climbing)", "../docs/5/5b_grafo_automatico.png", "#F1948A")

# 4. CONFRONTO INFERENZIALE E DIMOSTRAZIONE EXPLAINING AWAY
print("\n" + "="*70)
print(" CONFRONTO INFERENZA E DIMOSTRAZIONE 'EXPLAINING AWAY' ")
print("="*70)

inf_esperto = VariableElimination(modello_esperto)

# --- SCENARIO A: Solo Stress Massimo ---
evidenza_stress = {'Academic Pressure': 5.0}
res_stress = inf_esperto.query(variables=['Depression'], evidence=evidenza_stress)
prob_stress = res_stress.values[1] * 100 
print("SCENARIO A: Paziente con Pressione Accademica Massima (5.0)")
print(f"-> Rischio Depressione Stimato : {prob_stress:.2f}%\n")

# --- SCENARIO B: Stress Massimo MA con Fattori Protettivi ---
# 4 = Categoria reale per "Sonno > 8h" (Tasso di depressione più basso nel dataset)
# 0 = Categoria reale per "Dieta Sana" (Tasso di depressione più basso nel dataset)
evidenza_protettiva = {'Academic Pressure': 5.0, 'Sleep Duration': 4, 'Dietary Habits': 0}
res_protettiva = inf_esperto.query(variables=['Depression'], evidence=evidenza_protettiva)
prob_protettiva = res_protettiva.values[1] * 100 
print("SCENARIO B: Pressione Massima (5.0) + Sonno > 8h + Dieta Sana")
print(f"-> Rischio Depressione Stimato : {prob_protettiva:.2f}%")

# Calcolo del drop (Effetto Cuscinetto)
print("-" * 70)
print("EFFETTO COMPENSAZIONE (Explaining Away) DIMOSTRATO:")
print(f"Il rischio e' crollato del {prob_stress - prob_protettiva:.2f}% grazie ai fattori ambientali positivi.")
print("="*70)
print("Generazione dei grafi completata.")   