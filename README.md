# 'halp' Team Submission for USYD Challenge

## Install

To run our ensemble, simply run the `predict.py` file as shown below.

```bash
$ python predict.py /path/to/input/files
```

## Things we tried

### Optimizers we experimented with
- Stochastic Gradient Descent (with and without momentum)
- Adam
- Adagrad

Conclusion is that SGD typically converged to a more globally optimal solution than dynamic optimizers like Adam.

### Learning rate schedule we used

We experimented with constant learning rates ranging from 0.1 to 0.00001 as well as scheduled learning rates like the one below. 
We settled on the following learning rate schedule since our solution uses pre-trained models that just 

```python
def lr_schedule(epoch):
    lr = 0.0003
    if epoch > 25:
        lr = 0.00015
    if epoch > 30:
        lr = 0.000075
    if epoch > 35:
        lr = 0.00003
    if epoch > 40:
        lr = 0.00001
    return lr
```
