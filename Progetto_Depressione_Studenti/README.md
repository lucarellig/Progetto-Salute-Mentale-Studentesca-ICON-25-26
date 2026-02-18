# DSS Ibrido per la Salute Mentale Studentesca

Un Sistema di Supporto alle Decisioni (DSS) ibrido progettato per valutare, diagnosticare e spiegare il rischio clinico di depressione negli studenti universitari. 

Il progetto integra tecniche di **Machine Learning** (Random Forest), **Web Semantico** (Ontologie in OWL/HermiT), **Sistemi Esperti** (Prolog) e **Reti Bayesiane** (Analisi Eziologica e Causalità).

---

## Architettura del Progetto

Il repository segue le *best practice* dell'ingegneria del software, separando rigorosamente i dati, il codice sorgente e la documentazione:

* **`/data`**: Contiene il dataset originale (`Student Depression Dataset.csv`), i dataset preprocessati e arricchiti dalle inferenze logiche (`.csv`), e l'ontologia generata in formato standard W3C (`.owl`).
* **`/src`**: Contiene tutti gli script Python e le regole Prolog (`.pl`), orchestrati da una pipeline automatica.
* **`/docs`**: Contiene la documentazione ufficiale in PDF e tutti i grafici (matrici di confusione, curve PR, grafi bayesiani) generati dinamicamente dal codice.

---

## Mappatura: Relazione vs Codice Sorgente

Il flusso narrativo della **Relazione in PDF** segue un percorso didattico, mentre il **codice** segue i vincoli di dipendenza dei dati (Data Pipeline). Ecco dove trovare l'implementazione pratica di ogni capitolo:

| Capitolo PDF | Argomento | Script di Riferimento (`/src`) |
| :--- | :--- | :--- |
| **Fase Preliminare** | Pulizia dati ed EDA | `1_data_prep_eda.py` |
| **Argomento 1** | Machine Learning e XAI | `3_machine_learning_comparato.py` |
| **Argomento 2 (A)** | Ontologia e Reasoner (HermiT) | `2_ontologia_reasoning.py` |
| **Argomento 2 (B)** | Sistema Esperto (DSS in Prolog) | `4_integrazione_prolog.py` & `4_sistema_esperto.pl` |
| **Argomento 3** | Reti Bayesiane ed Explaining Away | `5_rete_bayesiana.py` |

---

## Requisiti e Installazione

>**(Nota per Windows: se il comando python non viene riconosciuto, utilizzare il launcher py digitando per esempio: py 0_run_pipeline.py)**

Il progetto richiede **Python 3.8+**. 
È fortemente raccomandato l'uso di un ambiente virtuale isolato (`venv` o `conda`) per evitare conflitti con altre librerie di sistema.

1. **Clona o estrai il progetto** in una cartella locale.
2. **Crea e attiva un ambiente virtuale** (opzionale ma consigliato):
   - Su Mac/Linux: `python -m venv venv` poi `source venv/bin/activate`
   - Su Windows: `python -m venv venv` poi `venv\Scripts\activate`
3. **Installa le dipendenze necessarie** eseguendo questo comando dalla directory principale del progetto:
   `pip install -r requirements.txt`
   
>**(Nota per Windows: se il comando pip non viene riconosciuto, utilizzare: py -m pip install -r requirements.txt)**

> **Nota per l'esecuzione del DSS in Prolog:** La libreria `pyswip` richiede che il software open-source **SWI-Prolog** sia installato nel sistema operativo e aggiunto alla variabile d'ambiente PATH.

---

## Come eseguire la Pipeline

L'intero progetto è stato ingegnerizzato per essere eseguito in modo sequenziale tramite uno script orchestratore. Non è necessario avviare i file uno alla volta.

1. Apri il terminale nella cartella principale del progetto.
2. Entra nella cartella dei sorgenti:
   
   `cd src`

3. ed esegui la pipeline con questo comando:

   `python 0_run_pipeline.py`

Il sistema eseguirà automaticamente le seguenti operazioni:
* **Fase 1:** Preprocessing, pulizia dati ed EDA.
* **Fase 2:** Costruzione dell'Ontologia in RAM, ragionamento tramite HermiT e arricchimento del dataset.
* **Fase 3:** Addestramento Machine Learning (Random Forest) e calcolo metriche XAI.
* **Fase 4:** Valutazione del Sistema Esperto in Prolog (incluso il rilevamento del *Falso Negativo* tramite Protocollo DELTA).
* **Fase 5:** Calcolo eziologico dei DAG Bayesiani ed esplorazione dell'effetto *Explaining Away*.


Tutti i dataset intermedi verranno salvati in `/data`, mentre tutti gli output grafici e visivi verranno depositati ordinatamente in `/docs`.


