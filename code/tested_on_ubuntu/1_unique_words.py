import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

"""
Extract unique words from intial dataset csv
"""

data_path = "../GSL_iso_files"
typename = str(sys.argv[1])
init_csv = pd.read_csv(os.path.join(data_path, typename+"_greek_iso.csv"), header=None)

init_csv = pd.DataFrame(init_csv[0].str.split("|",1).tolist())

to_csv = {}
x = init_csv[1].unique().tolist()
y = []
for word in x:
    pathlist = init_csv[init_csv[1] == word][0].tolist()
    y.append(len(pathlist))

x = [i for _, i in sorted(zip(y, x))]
y.sort()
x.reverse()
y.reverse()
x = x[:10]
y = y[:10]

for word in x:
    to_csv[word] = pd.Series(init_csv[init_csv[1] == word][0].tolist())

del init_csv
to_csv = pd.DataFrame(data=to_csv)
to_csv.to_csv(typename+'_unique.csv', index=False)

x = np.array(x)
y = np.array(y)

plt.figure()
plt.bar(x,y)
plt.xticks(rotation = 45)
plt.savefig(typename+'.png', bbox_inches='tight', pad_inches=0.3)