import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2,3, figsize=(12,8), sharey=True)

ax[0,0].hist(np.load('histogram1.npy'), 17)