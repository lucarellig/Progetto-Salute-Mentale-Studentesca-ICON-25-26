% ==============================================================================
% KBS: SISTEMA DI SUPPORTO ALLE DECISIONI (DSS) - PROTOCOLLI UNIVERSITARI
% ==============================================================================

% --- REGOLE DI INTERVENTO (PROTOCOLLI UFFICIALI) ---

% PROTOCOLLO ALFA (Emergenza Assoluta): Il ML rileva depressione E l'Ontologia rileva uno stato critico.
protocollo_intervento(ID, 'PROTOCOLLO ALFA: Rischio severo. Attivazione immediata sportello psicologico e tutorato accademico.') :- 
    rischio_ml(ID, alto), 
    rischio_onto(ID, critico).

% PROTOCOLLO BETA (Supporto Economico): Il ML rileva depressione + forte stress finanziario (ma no criticità ontologiche complesse).
protocollo_intervento(ID, 'PROTOCOLLO BETA: Rischio clinico associato a disagio economico. Indirizzare bando DSU (Diritto allo Studio).') :- 
    rischio_ml(ID, alto), 
    stress_finanziario(ID, alto),
    \+ rischio_onto(ID, critico).

% PROTOCOLLO GAMMA (Monitoraggio Standard): Il ML rileva depressione, ma senza aggravanti esterne evidenti.
protocollo_intervento(ID, 'PROTOCOLLO GAMMA: Rischio clinico isolato. Inserimento in lista di monitoraggio e invio questionario screening.') :- 
    rischio_ml(ID, alto), 
    \+ stress_finanziario(ID, alto),
    \+ rischio_onto(ID, critico).

% PROTOCOLLO DELTA (Prevenzione / Falso Negativo ML): Il ML non rileva depressione, MA l'Ontologia trova una combinazione critica.
protocollo_intervento(ID, 'PROTOCOLLO DELTA: Rischio latente rilevato dall''Ontologia (Falso Negativo ML). Consigliato colloquio preventivo.') :- 
    rischio_ml(ID, basso), 
    rischio_onto(ID, critico).

% PROTOCOLLO OMEGA (Nessun Intervento): Studente sano per il ML e sicuro per l'Ontologia.
protocollo_intervento(ID, 'PROTOCOLLO OMEGA: Nessuna criticita. Invio newsletter standard su benessere studentesco e gestione del sonno.') :- 
    rischio_ml(ID, basso), 
    \+ rischio_onto(ID, critico).