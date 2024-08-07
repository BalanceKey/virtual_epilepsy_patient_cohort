# Trying out some PCA
#check number of dominant directions

from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
pca_all = PCA(n_components=len(ch_names_sim))
pca_all.fit(y_sim.T)

npc=np.argmin(np.gradient(pca_all.explained_variance_ratio_))+1
plt.plot(pca_all.explained_variance_ratio_,'.')
plt.axvline(npc,color='k',alpha=0.5)
plt.show()

comp_sim=pca_all.components_
plt.plot(comp_sim[0])
plt.xticks(np.r_[1:len(ch_names_sim) + 1], ch_names_sim, fontsize=6, rotation=90)
plt.xlim([0, len(ch_names_sim) + 1])
plt.show()

# ## TODO eraseme
# from scipy.ndimage import gaussian_filter
# import matplotlib.cm as cm
# def myplot(x, y, s, bins=1000):
#     heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
#     heatmap = gaussian_filter(heatmap, sigma=s)
#     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#     return heatmap.T, extent
# r = y_sim.T
# x=np.dot(r,comp_sim[0])
# y=np.dot(r,comp_sim[1])
# # coarse graining of manifold space
# s = 8
# nbins=128
# img, extent = myplot(x, y, s, bins=nbins)
#
# # Manifold
# plt.figure(figsize=(12,4))
# plt.subplot(121)
# plt.title('Manifold Trajectory')
# plt.plot(x,y)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.subplot(122)
# plt.title('Manifold Density')
# plt.imshow(img, extent=extent, origin='lower', cmap=cm.jet,aspect='auto')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.tight_layout()
# plt.show()
#
# #########

pca_all = PCA(n_components=len(ch_names))
pca_all.fit(y.T)

npc=np.argmin(np.gradient(pca_all.explained_variance_ratio_))+1
plt.plot(pca_all.explained_variance_ratio_,'.')
plt.axvline(npc,color='k',alpha=0.5)
plt.show()

comp=pca_all.components_
plt.plot(comp[0])
plt.xticks(np.r_[1:len(ch_names) + 1], ch_names, fontsize=6, rotation=90)
plt.xlim([0, len(ch_names) + 1])
plt.show()

## comparing

sim_comp = comp_sim[0]
emp_comp = comp[0]
comparison = []
for ch in ch_names_sim:
    idx = ch_names.index(ch)
    comparison.append(emp_comp[idx])

plt.figure()
plt.plot(np.absolute(comparison), sim_comp, '.')
# plt.xlim([0, 0.3])
# plt.ylim([0, 0.3])
plt.show()


def calc_correlation(actual, predic):
    a_diff = actual - np.mean(actual)
    p_diff = predic - np.mean(predic)
    numerator = np.sum(a_diff * p_diff)
    denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2))
    return numerator / denominator
res = calc_correlation(np.absolute(comparison) / sum(np.absolute(comparison)), sim_comp / sim_comp.sum())


