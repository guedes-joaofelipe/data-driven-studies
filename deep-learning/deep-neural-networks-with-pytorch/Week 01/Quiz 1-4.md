### Simple Dataset

1. Which of the following is the correct way to compose transforms?

- 
```python
Compose([add_mult(dataset), mult(dataset)])
```


- >This
```python
transforms.Compose([add_mult(), mult()]
```


- 
```python
dataset.Compose([add_mult(), mult()])
```

2. What methods do you need in your data set class

- >This
```python
__init__,__getitem__ and __len__
```

- 
```python
__init__ and  __call__
```


