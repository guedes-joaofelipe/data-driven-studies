# Deeper Neural Networks : nn.ModuleList()

1. Consider the constructor for the following neural network class :

```python
class Net(nn.Module):
    # Section 1:
	def __init__(self, Layers):
		super(Net,self).__init__()
		self.hidden = nn.ModuleList()
		for input_size,output_size in zip(Layers,Layers[1:]):
			self.hidden.append(nn.Linear(input_size,output_size))

```

Let us create an object ```model = Net([2,3,4,4])```

How many hidden layers are there in this model?


Preview will appear here...

Enter math expression here

>2

2. Consider the forward function ,  fill out  the value for the if statement marked BLANK .

```python
# Section 2:
	def forward(self, activation):
		L=len(self.hidden)
		for (l, linear_transform) in zip(range(L), self.hidden):
			if #BLANK
				activation = torch.relu(linear_transform(activation))
			else:
				activation = linear_transform(activation)
		return activation
```

- l>L
- l > L-1
- >l<L-1


3. True or False  we use the following Class  or . Module for classification :

```python
class Net(nn.Module):

    # Constructor
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = torch.relu(linear_transform(activation))
            else:
                activation = torch.relu(linear_transform(activation))
        return activation
```

- >false
- true




## Dropout

1. In what situation would you use dropout for classification

- your training accuracy is the same as test accuracy
- >your training accuracy is much larger then your test accuracy


2. Consider the tensor x and the neural network object model that uses dropout . How would you make a prediction after training ?

- model.eval(); yhat=model(x)
- >model.train(); yhat=model(x)
- yhat=model(x)


3. Select the constructer value to let 40% of the  activations to the  shut off

- >nn.Dropout(0.4)
- nn.Dropout(0.7)


## Neural Network initialization

1. Select the best initialization for a linear object that will use the Relu  activation function .

- >This
```python
    linear=nn.Linear(input_size,output_size)
    torch.nn.init.kaiming_uniform_(linear.weight,nonlinearity='relu')
```
-
```python
    linear=nn.Linear(input_size,output_size
```

-
```python
linear=nn.Linear(input_size,output_sizetorch.nn.init.xavier_uniform_(linear.weight)
torch.nn.init.xavier_uniform_(linear.weight)
```


2. What type of initialisation method should you use for Relu


- default
- >He initialization
- Xavier initialization


## Batch Normalization

1. What task does Batch normalization do?

- >We normalize the input layer by adjusting and scaling the activations
- >Reducing Internal Covariate Shift


2. Consider  the following Batch Norm constructor. What is the parameter n_hidden1 represent

```python
nn.BatchNorm1d(n_hidden1)
```

- >the size of the input
- the input activation