import pandas as pd
from owlready2 import *

print("Avvio costruzione Ontologia e Ragionamento Logico (ML + OntoBK)...")

# ==========================================
# 1. CARICAMENTO DATI E CAMPIONAMENTO (Ottimizzazione!)
# ==========================================
df_completo = pd.read_csv("../data/dataset_pulito.csv")

# Estraiamo un campione di 2000 studenti per evitare l'esplosione combinatoria del Reasoner
df = df_completo.sample(n=2000, random_state=42).copy()
df.reset_index(drop=True, inplace=True) # Resettiamo l'indice da 0 a 1999

print(f"Dataset originale: {len(df_completo)} righe.")
print(f"Campione per il Reasoner (per limiti di complessità EXPTIME): {len(df)} righe.\n")

# ==========================================
# 2. DEFINIZIONE DELL'ONTOLOGIA (OWL)
# ==========================================
onto = get_ontology("http://progetto.uniba.it/student_depression.owl")

with onto:
    class Studente(Thing):
        pass
    
    class StudenteAdAltoRischio(Studente):
        pass

    class pressione_accademica(DataProperty, FunctionalProperty):
        domain = [Studente]
        range = [int]
        
    class stress_finanziario(DataProperty, FunctionalProperty):
        domain = [Studente]
        range = [float]

    # Regola Logica (OR)
    StudenteAdAltoRischio.equivalent_to = [
        Studente & 
        (pressione_accademica.value(4) | pressione_accademica.value(5)) & 
        (stress_finanziario.value(4.0) | stress_finanziario.value(5.0))
    ]

# ==========================================
# 3. POPOLAMENTO DELLA KNOWLEDGE BASE
# ==========================================
print("Popolamento della Knowledge Base in corso...")
individui_studenti = {}

for index, row in df.iterrows():
    nome_studente = f"studente_{index}"
    nuovo_studente = Studente(nome_studente)
    
    nuovo_studente.pressione_accademica = int(row['Academic Pressure'])
    nuovo_studente.stress_finanziario = float(round(row['Financial Stress']))
    
    individui_studenti[nome_studente] = nuovo_studente

# ==========================================
# 4. RAGIONAMENTO AUTOMATICO (REASONING)
# ==========================================
print("Avvio del Reasoner HermiT (motore inferenziale)...")
with onto:
    sync_reasoner(infer_property_values=True)

print("Ragionamento completato con successo in pochi secondi!\n")

# ==========================================
# 5. ESTRAZIONE E ARRICCHIMENTO DEL DATASET
# ==========================================
studenti_a_rischio_inferiti = set(onto.StudenteAdAltoRischio.instances())

nuova_feature = []
for index in range(len(df)):
    nome = f"studente_{index}"
    if individui_studenti[nome] in studenti_a_rischio_inferiti:
        nuova_feature.append(1)
    else:
        nuova_feature.append(0)

df['Onto_Rischio_Alto'] = nuova_feature

# Salvataggio
df.to_csv("../data/dataset_arricchito.csv", index=False)
onto.save(file="../data/ontologia_studenti.owl", format="rdfxml")

print(f"Fatto! Il Reasoner ha individuato logicamente {sum(nuova_feature)} studenti ad altissimo rischio su 2000.")
print("File 'dataset_arricchito.csv' e 'ontologia_studenti.owl' creati con successo!")