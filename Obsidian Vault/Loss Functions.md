
#### Sigmoid
Normalize x between 0 and 1
```
@staticmethod  
def Sigmoid(x:float):  
    return 1 / (1+math.e**-x)  
```
![[Sigmoid_example.png]]
#### ReLU
Min/Max x between 0 and 1
```
@staticmethod  
def ReLU(x:float):  
    return np.maximum(0, x)  
    #return max(0, min(x,1))  
```
![[ReLU_example.png]]
#### ReLUd
0 or 1
```
@staticmethod  
def ReLUd(x:float):  
    if x <= 0:  
        return 0  
 else:  
        return 1  
```
![[ReLUd_example.png]]