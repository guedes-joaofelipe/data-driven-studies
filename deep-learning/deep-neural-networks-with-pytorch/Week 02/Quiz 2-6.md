### Training Parameters in PyTorch

1. What does the following line of code do :

```python
w.grad.data.zero_()
```

- update  parameters
- >zero the gradients before running the backward pass
- calculate the iteration

2. What does the following line of code do :

```python
loss.backward()
```

- update  parameters
- >compute gradient of the loss with respect to all the learnable parameters
- zero the gradients before running the backward pass

