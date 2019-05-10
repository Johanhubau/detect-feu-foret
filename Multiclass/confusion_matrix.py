
#Modification to make : when only zeros in one line, not to divide it
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
array =  [[39, 0,0,  1,  0],
 [0, 34, 0, 0, 0],
 [ 3,  2, 25,  1,  7],
 [ 1,  1,  0,  5,  0],
 [11,  1, 6, 0,22]]

array = np.array(array)
array = (array.T/np.sum(array, axis=1)).T
df_cm = pd.DataFrame(array, ["Fire","Fog","Not Fire","Red object","Smoke"],
                  ["Fire","Fog","Not Fire","Red object","Smoke"])
plt.figure(figsize = (10,8))
sn.set(font_scale=1.4)#for label size
ax = sn.heatmap(df_cm, annot=True,annot_kws={"size": 16,})# font size
plt.xlabel('Classe prédite')
plt.ylabel('Vraie classe')
plt.show()
