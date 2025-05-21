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
python main.py -config configs/config.yaml -device cpu
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

# TODO:
Metrics:
- [ ] AHC
- [X] Robustness
- [X] OOD Detection

New settings:
- [X] small datasets
- [ ] ProtoPnet

New loss functions:
In the paper of parametric prototypes there are different loss functions to be tested. 
They concluded that the distance-based cross entropy loss (the one implemented) is the best.
However, maybe it is also interesting to implement the others in our case. 
I think this might strghten the paper justification, since we are emulating also the other metrics,
but we can discuss about this.

Tuning of the temperature:
Important! The temperature parameter has shown to be important in many situations. We shhould perform an ablation study on this parameter and put in the final table the best result for each geometry (and dataset of course).


Experiments:
- Resnet18 
- 3 seeds
- if small datasets --> pre-trained (No idea on the hyperparameters)

Experiment table:
| Seed | Cifar-10 | Cifar-100 | Aircraft | Cars | CUB |
| ----- |-----|-----|-----|-----|-----|
| 42    | Nan | Nan | Nan | Nan | Nan |
| 117   | Nan | Nan | Nan | Nan | Nan |
| 12345 | Nan | Nan | Nan | Nan | Nan |

