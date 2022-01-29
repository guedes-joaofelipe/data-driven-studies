### Mini-Batch Gradient Descent

1. You have 100 samples of data and your batch size is 50. How many iterations will it take to go through 1 epoch?

$iterations = \frac{samples}{batchSize} = \frac{100}{50} = 2$

> 2

2. Consider the dataset class Data(). How would you create a data loader object trainloader with a batch size of 3?

- >This
```python
data_set=Data()
trainloader=DataLoader(dataset=data_set,batch_size=3)
```

-
```python
data_set=Data(batch_size=3)
trainloader=DataLoader(dataset=data_set)
```