from pyswip import Prolog
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

print("="*70)
print(" FASE 4: DECISION SUPPORT SYSTEM (DSS) - PROTOCOLLI DI INTERVENTO")
print("="*70)

# 1. CARICAMENTO DATI E PREDIZIONE ML (La parte Statistica)
print("-> Addestramento rapido ML e classificazione degli studenti...")
df = pd.read_csv("../data/dataset_arricchito.csv")

X = df.drop(columns=['Depression'])
y = df['Depression']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

modello = LogisticRegression(random_state=42)
modello.fit(X_scaled, y)
# Aggiungiamo la predizione reale del ML per ogni studente
df['Predizione_ML'] = modello.predict(X_scaled) 

# 2. INIZIALIZZAZIONE PROLOG (La parte Logica)
print("-> Inizializzazione motore logico Prolog...")
prolog = Prolog()
prolog.consult("4_sistema_esperto.pl")

# 3. CAMPIONAMENTO E TRADUZIONE IN FATTI LOGICI
# Per dimostrare l'efficacia del DSS, estraiamo dal dataset reale 
# un campione rappresentativo per ogni singola casistica clinica.

campioni = []

# Cerchiamo 1 candidato ALFA (Rischio ML Alto + Ontologia Critica)
cand_alfa = df[(df['Predizione_ML'] == 1) & (df['Onto_Rischio_Alto'] == 1)]
if not cand_alfa.empty: campioni.append(cand_alfa.sample(1, random_state=42))

# Cerchiamo 1 candidato BETA (Rischio ML Alto + Stress Finanziario Alto + Ontologia Sicura)
cand_beta = df[(df['Predizione_ML'] == 1) & (df['Financial Stress'] >= 4) & (df['Onto_Rischio_Alto'] == 0)]
if not cand_beta.empty: campioni.append(cand_beta.sample(1, random_state=42))

# Cerchiamo 1 candidato GAMMA (Rischio ML Alto + Stress Fin Basso + Ontologia Sicura)
cand_gamma = df[(df['Predizione_ML'] == 1) & (df['Financial Stress'] < 4) & (df['Onto_Rischio_Alto'] == 0)]
if not cand_gamma.empty: campioni.append(cand_gamma.sample(1, random_state=42))

# Cerchiamo 1 candidato DELTA (Il Falso Negativo: ML Basso ma Ontologia Critica)
cand_delta = df[(df['Predizione_ML'] == 0) & (df['Onto_Rischio_Alto'] == 1)]
if not cand_delta.empty: campioni.append(cand_delta.sample(1, random_state=42))

# Cerchiamo 1 candidato OMEGA (Tutto sano)
cand_omega = df[(df['Predizione_ML'] == 0) & (df['Onto_Rischio_Alto'] == 0)]
if not cand_omega.empty: campioni.append(cand_omega.sample(1, random_state=42))

# Uniamo i campioni reali trovati
studenti_reali = pd.concat(campioni).reset_index()

print("-> Iniezione dei profili clinici nel Motore Inferenziale...\n")
for index, row in studenti_reali.iterrows():
    id_stud = int(row['index']) # L'indice originale della riga
    
    # Traduciamo le variabili per il Prolog
    ml_val = 'alto' if row['Predizione_ML'] == 1 else 'basso'
    onto_val = 'critico' if row['Onto_Rischio_Alto'] == 1 else 'sicuro'
    stress_val = 'alto' if row['Financial Stress'] >= 4 else 'basso'
    
    # Asseriamo i fatti
    prolog.assertz(f"rischio_ml({id_stud}, {ml_val})")
    prolog.assertz(f"rischio_onto({id_stud}, {onto_val})")
    prolog.assertz(f"stress_finanziario({id_stud}, {stress_val})")

# 4. ESTRAZIONE DEI PROTOCOLLI (Output del Sistema Esperto)
print("-" * 70)
print(f"{'ID PAZIENTE':<13} | {'PROFILO (ML / Ontologia / Stress)':<35} | {'DECISIONE DSS'}")
print("-" * 70)

for index, row in studenti_reali.iterrows():
    id_stud = int(row['index'])
    
    # Interroghiamo il Prolog per sapere quale Protocollo applicare
    protocollo_query = list(prolog.query(f"protocollo_intervento({id_stud}, Protocollo)"))
    
    if protocollo_query:
        protocollo = protocollo_query[0]['Protocollo']
        
        # Formattazione per la stampa
        ml_t = 'Alto' if row['Predizione_ML'] == 1 else 'Basso'
        onto_t = 'Critico' if row['Onto_Rischio_Alto'] == 1 else 'Sicuro'
        stress_t = 'Alto' if row['Financial Stress'] >= 4 else 'Basso'
        
        profilo = f"ML:{ml_t} / Onto:{onto_t} / Fin:{stress_t}"
        
        # Stampiamo il protocollo dividendolo prima e dopo i due punti per formattarlo bene
        nome_prot, desc_prot = protocollo.split(': ', 1)
        
        print(f"Studente #{id_stud:<4} | {profilo:<35} | {nome_prot}")
        print(f"{'':<13} | {'':<35} | -> {desc_prot}\n")

print("-" * 70)
print("Esecuzione Fase 4 completata. Interventi prescritti con successo.")