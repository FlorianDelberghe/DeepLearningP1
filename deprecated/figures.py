# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

#%% Accuracy with seed

logs = [file for file in os.listdir('figs') if file.endswith('.log')]

class_err = {}
comp_err = {}

for filename in logs:

    class_err[filename[:-11]] = []
    comp_err[filename[:-11]] = []

    with open(os.path.join('figs', filename), mode='rt') as f:
        for i, line in enumerate(f):
            class_err[filename[:-11]].append(float(line.split()[2][:-2]))
            comp_err[filename[:-11]].append(float(line.split()[5][:-1]))

        class_err[filename[:-11]], comp_err[filename[:-11]] = np.asarray(class_err[filename[:-11]]), np.asarray(comp_err[filename[:-11]])

fig, ax = plt.subplots()

name = ['compNet2', 'compNet4', 'compNet5']
for i in [0, 1, 2]:
    ax.bar([(i+1)-0.3], [class_err[name[i]+'fc'].mean()], width=0.2, color='#1f77b4',
            yerr=class_err[name[i]+'fc'].std(), align='center', ecolor='black', capsize=6)
    ax.bar([(i+1)], [comp_err[name[i]+'fc'].mean()], width=0.2, color='#ff7f0e',
            yerr=comp_err[name[i]+'fc'].std(), align='center', ecolor='black', capsize=6)
    ax.bar([(i+1)+0.2], [comp_err[name[i]+'naive'].mean()], width=0.2, color='#2ca02c',
            yerr=comp_err[name[i]+'naive'].std(), align='center', ecolor='black', capsize=6)

ax.set_xticks([1, 2, 3]); ax.set_xticklabels(name);
ax.set_ylabel('Error [%]')
ax.legend(['Classification', 'Net comparison', 'Naive comparison'], loc=2)
plt.savefig(os.path.join('figs', 'NetVariance.png'))
#fig.suptitle('Errors of the final nets with SD for different initializations')

#%%
