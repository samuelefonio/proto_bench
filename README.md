# Code implementation for the paper "Riemannian Optimization for Hyperbolic Prototypical Networks" 

This repository is released for reproducibility.

Before running the code, install the requirements.txt 

```
# Create a virtual environment
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Install the packages from requirements.txt
pip install -r requirements.txt
```

The configuration files in configs allow for easy and immediate reproducibility. In case a dataset among cars, cub2011 or aircraft is not downloaded yet, please run the respective python file. The most difficult dataset to get is the cars dataset, a useful link in the python file is provided.

Once the dataset is downloaded, the command to run an experiment is:
```
python main.py -config configs/config.json -device cpu
```

## quick guide
Ciao :smiley: In realta' la repo e' molto piu' semplice di quello che sembra. Un elemento fondamentale e' il logger. Il logger gestisce, appunto
il log dei file. Puo' essere di 2 tipi:   
- Log (locale)
- wandb

In entrambi i casi ti stampa qualcosa a video, e ci sono 2 modi per farlo:  
- logger(stats): dove stats e' un dizionario di statistiche, tipo {"training_accuracy": accuracy}. Importante! Se usi wandb devi mettere uno step, che puo' essere tipo l'epoca, boh, a piacimento
- logger.info(message): questo stampa un messaggio normale, in caso volessi printare un messaggio normale nel log ma senza statistiche.

Secondo me questa e' la cosa piu' complicata da capire a livello di codice ahahah  
Per quanto riguarda geometrie, metriche eccetera, si puo' fare riferimento a metric_model che ritorna una distanza in output a seconda della geometria che si e' scelto. Probabilmente non andremo a toccare nulla di tutto cio' ma vabbe.

DETTO CIO: Se vuoi usare questi tool topperia, consiglio probabilmente se vogliamo poi giocare con protopnet eccetera di implementarle in un altro file e creaare un main_protopnet.py dove avviene cio' che deve avvenire. Secondo me figo. Have fun!

## Sam TO DO:
New metrics:
- [] AHC
- [] Robustness
- [] OOD Detection

New settings:
- [] few-shot-learning
- [] small datasets

Possible alternatives:
- [] add hierarchical information
- [] implementing Euclidean Entailment Cones
