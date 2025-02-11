<p align="center">
  <img src="https://github.com/user-attachments/assets/cf38bdb0-e617-4988-a120-ef206062cea2" width=100% alt="logo">
</p>

# Carval

Carval è un sistema di Machine Learning sviluppato per supportare gli utenti 
nella valutazione del prezzo delle vetture automobilistiche di seconda mano.

L'obiettivo principale del progetto è fornire una valutazione data-driven 
del prezzo delle auto usate, aiutando sia i privati che le concessionarie a 
prendere decisioni informate.

## Autori

- Michele Antonio Annunziata - [micheleantonioannunziata](https://github.com/micheleantonioannunziata)
- Raffaele Coppola - [Raff2812](https://github.com/Raff2812)
- Domenico Cirillo - [DomenicoDC25](https://github.com/DomenicoDC25)
- Luigi Auriemma - [LuigiAuriemma](https://github.com/LuigiAuriemma)

## Struttura del progetto  

La repository di Carval è organizzata nelle seguenti directory:  

- **`dataset/`**. Contiene il dataset originale reperito da Kaggle ([qui](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices)) 
e il dataset pre-elaborato utilizzato per l’addestramento dei modelli.  
- **`diagrams/`**. Raccoglie tutti i grafici generati durante l'analisi esplorativa 
dei dati e la fase di sviluppo della pipeline di Machine Learning.  
- **`docs/`**. Include la documentazione e presentazione relativa al progetto.  
- **`scripts/`**: Contiene tutti gli script Python e Jupyter Notebook sviluppati.  

## Come utilizzare il progetto  

Per eseguire Carval, segui questi semplici passaggi:  

### 1. Installazione delle dipendenze  

Prima di avviare il sistema, è necessario installare tutte le dipendenze richieste. 
Assicurati di avere Python installato, quindi esegui il seguente comando dalla root del progetto:  

```bash
pip install -r requirements.txt
```

### 2. Esecuzione della pipeline

Una volta installate le dipendenze, esegui la pipeline di pre-elaborazione dei dati e addestramento del modello. 
Vai nella cartella scripts/pipeline e lancia lo script.

```bash
python scripts/pipeline/carval-pipeline.py
```

### 3. Avvio dell'interfaccia utente

Dopo aver completato la fase di preprocessing e addestramento, puoi eseguire l'interfaccia web per interagire con il modello.
Spostati nella directory scripts con il comando `cd scripts` ed esegui il seguente comando:

```bash
streamlit run app.py
```

Questo avvierà l'applicazione Streamlit, permettendoti di testare il sistema direttamente dal browser.

#### Piccola demo

Per vedere Carval in azione, guarda la demo dell'interfaccia utente nel video qui sotto:

https://github.com/user-attachments/assets/c4fd6734-3758-47c2-aa22-ba7d5f3dde59


