
#Modification to make : when only zeros in one line, not to divide it
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
array = [[21,1,0,1],
     [0,0,0,0],
     [9,3,7,2],
     [0,0,0,2]]
array = np.array(array)
array = (array.T/np.sum(array, axis=1)).T
df_cm = pd.DataFrame(array, ["Fire","Fog","Not Fire","Red object"],
                  ["Fire","Fog","Not Fire","Red object"])
plt.figure(figsize = (10,8))
sn.set(font_scale=1.4)#for label size
ax = sn.heatmap(df_cm, annot=True,annot_kws={"size": 16,})# font size
plt.xlabel('Classe prédite')
plt.ylabel('Vraie classe')
plt.show()
