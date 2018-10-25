# Neural model based rl

## Initial algorithm set

* [v] Transition Model Prediction + Multi Step A-star Planning

#### Running Experiments
training transition model network with random sample actions

```
python parta.py -s True
```
training transition model network with A star sample actions

```
python partb.py -s True -l True
```
testing

```
python partb.py -t True
```



