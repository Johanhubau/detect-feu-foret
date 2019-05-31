
#Modification to make : when only zeros in one line, not to divide it
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
array =  [[77, 44, 33,  6],
 [77, 44, 33, 6],
 [ 76,  45, 33,  6],
 [ 77,  44,  33,  6],
 ]

array = np.array(array)
array = (array.T/np.sum(array, axis=1)).T
df_cm = pd.DataFrame(array, ["Fire","Smoke","Not Fire","Red object"],
                  ["Fire","Smoke","Not Fire","Red object"])
plt.figure(figsize = (10,8))
sn.set(font_scale=1.4)#for label size
ax = sn.heatmap(df_cm, annot=True,annot_kws={"size": 16,})# font size
plt.xlabel('Classe prédite')
plt.ylabel('Vraie classe')
plt.show()
