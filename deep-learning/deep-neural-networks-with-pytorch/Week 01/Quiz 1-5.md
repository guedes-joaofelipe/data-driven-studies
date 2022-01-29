### Datasets

1. Consider the following lines of code from the lab:

```python
dataset = Dataset(csv_file=csv_file, data_dir=directory)
a=dataset[0][0]
b=dataset[0][1]

```
what variable  contains the image ?

- >a
- b


2. consider the following lines of code from the lab:

```python
dataset = Dataset(csv_file=csv_file, data_dir=directory)
```

How would you obtain the tenth sample's label:


- >This
```python
dataset[9][1]
```

-
```python
dataset[9][0]
```