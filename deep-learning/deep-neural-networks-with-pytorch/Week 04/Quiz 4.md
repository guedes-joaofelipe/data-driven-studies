## Softmax Function

1. How would you classify the purple point given the three lines used in a softmax classifier:

![](6-1-1.png)

- yhat=0 or blue 
- >yhat=1 or red
- yhat=2 or green


2. Consider the following output of the lines used in the softmax function shown in the following table. What will be the value of yhat ?

![](6-1-2.png)

- yhat=0
- yhat=1
- >yhat=2


## Softmax Prediction

1. Consider the following lines of code, what is yhat?

```python
    z = torch.tensor([[2,5,0],[10,8,2],[6,5,1]])
    _, yhat = z.max(1)
```

- >This
```python
tensor([1,0,0])
```
- 
```python
tensor([5,10,5])
```

- 
```python
tensor([1,1,1])
```

2. We have two input features and four classes , what are the parameters for Softmax() constructor according to the above code?

```python
class Softmax (nn.Module):

    def __init__(self, in_size, out_size):

        super(Softmax, self).__init__()

        self.linear=nn.Linear(in_size, out_size)

    def forward(self, x):
```

- Sofmax(4,2)
- >Sofmax(2,4)
- Sofmax(4,4)


## Softmax PyTorch Quizz

1. What is the task of the following line of code?

```python
    transforms.ToTensor()
```

- Delete a tensor 
- Create a new tensor 
- >Convert the image to a tensor 


2. You have a 10x10 image and you would like to convert it to a vector or a rank one tensor, how many elements does it have?

> 100