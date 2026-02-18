import subprocess
import time

# Lista ordinata dei file da eseguire
scripts = [
    "1_data_prep_eda.py",
    "2_ontologia_reasoning.py",
    "3_machine_learning_comparato.py",
    "4_integrazione_prolog.py",
    "5_rete_bayesiana.py"
]

print("="*70)
print(" AVVIO PIPELINE: KNOWLEDGE-BASED SYSTEM (Student Depression) ")
print("="*70)

start_total = time.time()

for script in scripts:
    print(f"\n\n{'='*50}")
    print(f" ESECUZIONE MODULO: {script}")
    print(f"{'='*50}\n")
    
    # Esegue lo script Python e aspetta che finisca prima di passare al successivo
    subprocess.run(["python", script])

end_total = time.time()
tempo_totale = round(end_total - start_total, 2)

print("\n\n" + "="*70)
print(f" PIPELINE COMPLETATA CON SUCCESSO IN {tempo_totale} SECONDI! ")
print(" Tutti i grafici e i modelli sono stati generati nella directory.")
print("="*70)