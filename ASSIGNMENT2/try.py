
import pandas as pd
import numpy as np
 
t1 = pd.DataFrame(np.arange(12).reshape(3, 4), index=list('abc'), columns=list('wxyz'))
print(t1)
 

# 取第a行,和第c行[[],[]],取连续范围[,]
ac = t1.loc[['a', 'c'], :]
acc = t1.loc['b': 'c', :]
print(ac)
print(acc)
#iloc
iac = t1.iloc[[2,1], :]
iacc = t1.iloc[:,1:2]
print(iac)
print(iacc)