import pandas as pd
import numpy as np

data=pd.read_csv('2.csv')
rows=data.shape[0]
cols=data.shape[1]-1
concepts = np.array(data.iloc[:,0:-1])
target = np.array(data.iloc[:,-1])
spec_h=list()
for i in range(rows):
    if data.iloc[i,cols]=='Yes':
        for j in data.iloc[i]:
            spec_h.append(j)
        break
spec_h.pop()
gen_h = [["?" for i in range(cols)] for i in range(cols)]
for i, h in enumerate(concepts):
    if target[i] == "Yes":
        for x in range(cols):
            if h[x] != spec_h[x]:
                spec_h[x] = '?'
                gen_h[x][x] = '?'
    if target[i] == "No":
        for x in range(cols):
            if h[x] != spec_h[x]:
                gen_h[x][x] = spec_h[x]
            else:
                gen_h[x][x] = '?'
indices = [i for i, val in enumerate(gen_h) if val == ['?', '?', '?', '?', '?', '?']]
for i in indices:
    gen_h.remove(['?', '?', '?', '?', '?', '?'])
print("Final Specific_h:", spec_h, sep="\n")
print("Final General_h:", gen_h, sep="\n")