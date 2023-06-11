import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import pandas as pd
import matplotlib as mpl
import matplotlib.patches as mpatches

def hist_subplot(data1,data2,xlabel1,xlabel2,title1,title2,plot_median = False):
  """
  Create a figure with two subplots displaying distribution plots.
  """
  if plot_median:
    median1 = data1.median()
    median2 = data2.median()
  fig, axs = plt.subplots(1, 2, figsize=(12, 6))
  sns.histplot(data1, kde=True, bins=30, ax=axs[0])
  axs[0].set_title(title1)
  if plot_median:
    axs[0].axvline(x=median1, color='red', linestyle='--', label='Median')
    axs[0].legend()
  axs[0].grid()
  axs[0].set_xlabel(xlabel1)
  sns.histplot(data2, kde=True, bins=30, ax=axs[1])
  axs[1].set_title(title2)
  if plot_median:
    axs[1].axvline(x=median2, color='red', linestyle='--', label='Median')
    axs[1].legend()
  axs[1].grid()
  axs[1].set_xlabel(xlabel2)
  plt.tight_layout()
  fig.show()


def plot_bic(components,values,idx,criterion):
  """
  Create a figure of the BIC values per number of GMM components
  """
  plt.figure(figsize=(8,8))
  sns.lineplot(x = np.arange(1,components+1,1), y =values, markers=".")
  plt.axvline(x = idx, color = 'red',label = "Optimal number of components",linestyle="dashed" )
  plt.title(f"{criterion} values per # of components")
  plt.xlabel(f"Number of components")
  plt.xticks(np.arange(1,components+1,1))
  plt.ylabel(f"{criterion} score")
  plt.legend()
  plt.show()

def plot_pca(adata,title = "PCA", color = None):
  """
  Create a figure of the the two PCs after pca preprocessing on the adata object using scanpy
  """
  plt.figure(figsize=(8,8))
  plt.title(title)
  if color is not None:
    plt.scatter(x = adata.obsm['X_pca'][:,0],y = adata.obsm['X_pca'][:,1],c = color, edgecolors='black', alpha= 0.8)
  else:
    plt.scatter(x = adata.obsm['X_pca'][:,0],y = adata.obsm['X_pca'][:,1], edgecolors='black', alpha= 0.8)
  plt.xlabel(f"PC1: {round(adata.uns['pca']['variance_ratio'][0]*100,2)}% of total variance")
  plt.ylabel(f"PC2: {round(adata.uns['pca']['variance_ratio'][1]*100,2)}% of total variance")
  plt.grid()
  plt.show()

def plot_pca_variance(variance_ratio,n_components,variance_cutoff = 0.90,verbose = True):
  cummulative_variance = np.cumsum(variance_ratio)
  plt.figure(figsize=(8,8))
  sns.lineplot(x = np.arange(1,n_components+1,1), y =cummulative_variance,marker='o')
  plt.xlabel("Principal Components")
  plt.axhline(y=variance_cutoff, color = 'red',label = "90% of the total variance",linestyle="dashed")
  plt.ylabel("Cummulative Variance explained")
  plt.title("CDF of the explained variance of the PCs")
  plt.yticks(np.arange(0,1.1,0.1))
  plt.legend()
  plt.grid()
  plt.show()
  #Show the PCs kept with this method
  if verbose:
    print(f"Variance Threshold of {variance_cutoff}% keeps: ",(cummulative_variance[cummulative_variance <= variance_cutoff]).shape[0], "PCs")

def make_ellipses(gmm, X, labels, title = "PCA"):
    num_components = len(gmm.means_)
    colors = plt.cm.get_cmap('tab10', num_components)
    fig, ax = plt.subplots(figsize = (7,7))
    for n in range(num_components):
        color = colors(n)
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color
        )
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")
    
    for i,mean in enumerate(gmm.means_):
        color = colors(i)
        data = X[labels == i]
        ax.scatter(data[:, 0], data[:, 1], s=50, linewidths=2,edgecolors ="black", color=color, label=f"Component {i}",alpha=0.8)
        ax.scatter(mean[0],mean[1], marker= 'x', s = 100, color = "black",linewidths = 4, edgecolors="white")
    plt.title(title)
    plt.legend()
    if title == "PCA":
      plt.xlabel("PC1")
      plt.ylabel("PC2")
    else:
      plt.xlabel(f"{title}1")
      plt.ylabel(f"{title}2") 
    plt.grid()
    plt.legend()
    plt.show()

def make_ellipses_joint(gmm, X,labels,proba, title = "PCA",cmap = "Blues",joint = False):
    """
    Visualizes a Gaussian Mixture Model (GMM) with ellipses representing the eigenvalues of the covariance matrices of the components.
    Additionally, it plots the data points with markers and colors based on the posterior probabilities.
    """
    markers = ['o', 'v', 'h', 's', '<', 'D', '1', 'H', '|', '2', '>', 'p', '_', 'x', '*', 'd', '+', 'h', ',', '^', '.']
    num_components = len(gmm.means_)
    colors = plt.cm.get_cmap('tab20b', num_components)
    cmap = plt.cm.get_cmap(cmap)
    fig, ax = plt.subplots(figsize = (15,15))
    for n in range(num_components):
        color = colors(n)
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color
        )
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")
    scatter = ax.scatter([], [], s=80, edgecolors='none', c=[], cmap=cmap, alpha=0.8)
    
    if joint:
      comp_porba = proba
      data = X
      ax.scatter(data[:, 0], data[:, 1], s= 80, edgecolors = "black",marker="o", linewidths = 1.5, c=comp_porba,alpha=0.8, cmap=cmap)
    else:
      for i,means in enumerate(gmm.means_):
        color = colors(i)
        comp_porba = proba[:,i][labels == i]
        data = X[labels == i]
        ax.scatter(data[:, 0], data[:, 1], s= 80, edgecolors = color,marker=markers[i], linewidths = 1.5, c=comp_porba, label=f"State {i}",alpha=0.8, cmap=cmap)
        # ax.scatter(mean[0],mean[1], marker= 'x', s = 90, color = "black",linewidths = 4, edgecolors="black")
    for j, row in enumerate(X): 
      x = row[0]
      y = row[1]
      cell_id = j+1
      ax.annotate(str(cell_id), (x, y), fontsize=12, fontweight='bold', ha='center', va='center',xytext=(0, -12), textcoords='offset points')
    cbar = plt.colorbar(scatter)
    if joint:
      cbar.set_label('Marginal Probability')
    else:
      cbar.set_label('State posterior Probability')
    plt.title(title)
    if title == "PCA":
      plt.xlabel("PC1")
      plt.ylabel("PC2")
    else:
      plt.xlabel(f"{title}1")
      plt.ylabel(f"{title}2") 
    plt.grid()
    plt.legend()
    plt.show()

def posterior_heatmap(posteriors,cmap = "bwr",c_label = 'Posterior Probability'):
  """
  visualization to represent the posterior probabilities of different cells belonging to specific states
  """
  df = pd.DataFrame(posteriors)
  col_list = list(df.columns)
  df = df.sort_values(ignore_index=False, by=col_list,axis=0)
  plt.figure(figsize=(20,5))
  sns.heatmap(df.T, yticklabels=True,cmap=cmap,cbar_kws={'label': c_label})
  plt.xlabel("Cell id")
  plt.xticks(rotation = 90)
  plt.title("Cell trajectories")
  plt.ylabel("Cell States")
  plt.show()



def plot_state_cellsums(posteriors, threshold=0.90, labels=None):
    cell_sums = np.zeros((posteriors.shape[1],))
    for state in range(posteriors.shape[1]):
        criterion_1 = np.max(posteriors, axis=1) > threshold
        criterion_2 = np.argmax(posteriors, axis=1) == state
        cell_sums[state] = (criterion_1 & criterion_2).sum()
    
    unknown = posteriors.shape[0] - cell_sums.sum()
    cell_sums = np.append(cell_sums, unknown)
    x_labels = list(np.arange(posteriors.shape[1] + 1))
    x_labels[-1] = 'Transitioning state'
    
    
    plt.figure(figsize=(10, 10))
    legend_patches = []
    if labels is not None:  
        true_cell_sums = np.zeros((posteriors.shape[1],))
        for i in range(len(labels)):
            state_label = labels[i]
            true_cell_sums[state_label] += 1
        true_cell_sums = np.append(true_cell_sums, 0)
        sns.barplot(x=x_labels, y=true_cell_sums, alpha=1.0, color='tab:red', dodge=True)
        legend_patches.append(mpatches.Patch(color = 'tab:red', label='Hard cell state counts'))
    
    sns.barplot(x=x_labels, y=cell_sums, color= "tab:blue",alpha = 0.8)
    plt.title(f"Cell counts per state, membership threshold probability > {threshold}")
    plt.grid()
    plt.ylabel("Cell counts")
    plt.xlabel("Cell States")
    legend_patches.append(mpatches.Patch(color='tab:blue', label='Cell states counts (Thresholded)',alpha = 0.8))
    plt.legend(handles=legend_patches)
    plt.show()


