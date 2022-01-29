## Two-Dimensional Tensors

1. How  do you  convert the following Pandas Dataframe to a tensor:

```python
df = pd.DataFrame({'A':[11, 33, 22],'B':[3, 3, 2]})
```

- >torch.tensor(df.values)
- torch.tensor(df)

2. What is the result  of the following:

```python
A = torch.tensor([[0, 1, 1], [1, 0, 1]])
B = torch.tensor([[1, 1], [1, 1], [-1, 1]])
A_times_B = torch.mm(A,B)
```

- >tensor([[0, 2], [0, 2]])
- tensor([[0, 1], [1, 4]])

