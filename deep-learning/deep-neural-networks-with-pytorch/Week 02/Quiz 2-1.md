### Prediction in One Dimension

1. What is wrong with the following lines of code:

```python
class LR():
    
    # Constructor
    def __init__(self, input_size, output_size):
        
        # Inherit from parent
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    # Prediction function
```

- >its  missing  nn.Module
- there is no call function 


2. What is wrong with the following lines of code:

```python
class LR(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        
        # Inherit from parent
        super(dog, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    # Prediction function
```

- >super(dog, self) should be  super(LR, self) 
- there is no call function 