import matplotlib.pyplot as plt

labels = ['1', '4', '8', '16']

c1k_decaf = [1174, 1115, 1095, 1218]
c1_yerr_decaf = [112, 137, 136, 114]
c1k_hog = [3727, 4274, 0, 0]
c1_yerr_hog = [288, 203, 0, 0]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
ax.bar(x - width/2, c1k_decaf, width, label='DECAF', yerr=c1_yerr_decaf)
ax.bar(x + width/2, c1k_hog, width, label='HOG', yerr=c1_yerr_hog)

ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()