### Optimization in PyTorch

1. What does the following line of code do?

```python
optimizer.step()
```

- >Makes an update to its parameters
- Makes a prediction 
- Clears the gradient 
- Computes the gradient of the loss with respect to all the learnable parameters


2. What's wrong with the following lines of code?

```python
optimizer = optim.SGD(model.parameters(), lr = 0.01)
model=linear_regression(1,1)
```

- >The model object has not been created. As such, the argument that specifies what Tensors should be optimized does not exist
- There is no loss function 
- You have to clear the gradient 


