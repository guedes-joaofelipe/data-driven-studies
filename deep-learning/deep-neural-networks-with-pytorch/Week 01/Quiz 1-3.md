### Derivatives in PyTorch

1 How would you determine the derivative of $ y = 2x^3+x $ at $x=1$


- >This
```python
x = torch.tensor(1.0, requires_grad=True)
y = 2 * x ** 3 + x
y.backward()
 x.grad
```

-
```python
x = torch.tensor(1.0, requires_grad=True)
y = 2 * x ** 3 + x
y.backward()
 y.grad
```

2. Try to determine partial derivative 𝑢u of the following function where $u=2$ and $v=1$: $f = uv + (uv)^2$

- >This
```python
u = torch.tensor(2.0, requires_grad = True)
v = torch.tensor(1.0, requires_grad = True)
f = u * v + (u * v) ** 2
f.backward()
print("The result is ", u.grad)
```

- 
```python
u = torch.tensor(2.0, requires_grad = True)
v = torch.tensor(1.0, requires_grad = True)
f = u * v + (u * v) ** 2
f.backward()
print("The result is ", v.grad)
```

