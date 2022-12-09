```python
import pandas as pd
import numpy as np
%run ./LSR_lib.ipynb
from numpy.random import seed
from numpy.random import normal
from sklearn.linear_model import LinearRegression
```


```python
#visualization libraries
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import statistics
```


```python
mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)

```


```python
#find the regression parameters a and b based on the X and Y you provide
x = [12,16,71,99,45,27,80,58,4,50]
y = [56,22,37,78,83,55,70,94,12,40]
find_reg_params(x, y)

```




    (31.82863092838909, 0.4950512786062967)




```python
regression(5, 20, 10, 1)
```




    (12.06049509306487, 0.8454579511267202)




```python
#number of simulations
n_sim=1000

#number of observations
obs=20

a=[]
b=[]


for i in range(1, n_sim+1):
    (a_val, b_val)=regression(5, 20000, 10, 1)
    a.append(a_val)
    b.append(b_val)
    
plt.hist(a, 100)
# plotting mean line for parameter a
plt.axvline(np.mean(a), color='k', linestyle='dashed', linewidth=2)
plt.show()

# plotting mean line for parameter b
plt.hist(b, 100)
plt.axvline(np.mean(b), color='k', linestyle='dashed', linewidth=2)
plt.show()

    
```


    
![png](output_5_0.png)
    



    
![png](output_5_1.png)
    



```python
#print the mean results for a and b
print ( "a and b are  ", np.mean(a), np.mean(b))
```

    a and b are   9.997416975203482 1.0000002517162234
    


```python

```
