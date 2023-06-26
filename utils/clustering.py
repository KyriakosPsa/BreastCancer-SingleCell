from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import fowlkes_mallows_score
import numpy as np
import scanpy as sc
import anndata as ad


def hierarchical(data,adata,linkage,n_clusters,ground_labels = None,comparison_key = "Ground labels",compare = True):
  model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage=linkage)
  labels = model.fit_predict(data)
  score = silhouette_score(data,labels)
  print(f"The silhouete score in the optimal PCA space is: {score}")
  if compare:
    ari = adjusted_rand_score(ground_labels, labels)
    ami = adjusted_mutual_info_score(ground_labels, labels)
    fow = fowlkes_mallows_score(ground_labels,labels)

    print(f"The adjusted random index (ARI) score with the original labels is: {ari}")
    print(f"The adjusted mutual information (AMI) score with to the original labels is: {ami}")
    print(f"The fowlkes mallows score with the original labels is: {fow}")

  adata.obs[f'{linkage} link Predictions'] = labels.astype(str)
  if compare:
    sc.pl.pca(adata, color=[f"{linkage} link Predictions",comparison_key],add_outline=True, size = 100, palette="coolwarm")
    sc.pl.umap(adata, color=[f"{linkage} link Predictions",comparison_key],add_outline=True, size = 100, palette="coolwarm")
    sc.pl.tsne(adata, color=[f"{linkage} link Predictions",comparison_key],add_outline=True, size = 100, palette="coolwarm")
    return score,ari,ami,fow
  else:
    sc.pl.pca(adata, color=[f"{linkage} link Predictions"],add_outline=True, size = 100, palette="coolwarm")
    sc.pl.umap(adata, color=[f"{linkage} link Predictions"],add_outline=True, size = 100, palette="coolwarm")
    sc.pl.tsne(adata, color=[f"{linkage} link Predictions"],add_outline=True, size = 100, palette="coolwarm")
    return score


def spectral(data,adata,n_clusters,n_neighbors,ground_labels = None,comparison_key = "Ground labels",compare = True):
  model = SpectralClustering(n_clusters=n_clusters,affinity='nearest_neighbors',n_neighbors=n_neighbors,assign_labels="cluster_qr",random_state=42)
  labels = model.fit_predict(data)
  score = silhouette_score(data,labels)
  print(f"The silhouete score in the optimal PCA space is: {score}")
  if compare:
    ari = adjusted_rand_score(ground_labels, labels)
    ami = adjusted_mutual_info_score(ground_labels, labels)
    fow = fowlkes_mallows_score(ground_labels,labels)
    print(F"The adjusted random index (ARI) score with the original labels is: {ari}")
    print(F"The adjusted mutual information (AMI) score with to the original labels is: {ami}")
    print(f"The fowlkes mallows score with the original labels is: {fow}")
  
  adata.obs['Spectral Predictions'] = labels.astype(str)
  if compare:
    sc.pl.pca(adata, color=["Spectral Predictions",comparison_key],add_outline=True, size = 100, palette="coolwarm")
    sc.pl.umap(adata, color=["Spectral Predictions",comparison_key],add_outline=True, size = 100, palette="coolwarm")
    sc.pl.tsne(adata, color=["Spectral Predictions",comparison_key],add_outline=True, size = 100, palette="coolwarm")
    return score,ari,ami,fow
  else:
    sc.pl.pca(adata, color=["Spectral Predictions"],add_outline=True, size = 100, palette="coolwarm")
    sc.pl.umap(adata, color=["Spectral Predictions"],add_outline=True, size = 100, palette="coolwarm")
    sc.pl.tsne(adata, color=["Spectral Predictions"],add_outline=True, size = 100, palette="coolwarm")
  return score


def GMM(data,adata,covariance_type,n_components,random_state, init_params,ground_labels=None,comparison_key = "Ground labels",compare = True):
  model = GaussianMixture(covariance_type=covariance_type,random_state = random_state,init_params=init_params,n_components=n_components)
  labels = model.fit_predict(data)
  score = silhouette_score(data,labels)
  print(f"The silhouete score in the optimal PCA space is: {score}")
  print(f"The silhouete score in the optimal PCA space is: {score}")
  if compare:
    ari = adjusted_rand_score(ground_labels, labels)
    ami = adjusted_mutual_info_score(ground_labels, labels)
    fow = fowlkes_mallows_score(ground_labels,labels)
    print(f"The adjusted random index (ARI) score with the original labels is: {ari}")
    print(f"The adjusted mutual information (AMI) score with to the original labels is: {ami}")
    print(f"The fowlkes mallows score with the original labels is: {fow}")
  adata.obs['GMM Predictions'] = labels.astype(str)
  if compare:
    sc.pl.pca(adata, color=["GMM Predictions",comparison_key],add_outline=True, size = 100, palette="coolwarm")
    sc.pl.umap(adata, color=["GMM Predictions",comparison_key],add_outline=True, size = 100, palette="coolwarm")
    sc.pl.tsne(adata, color=["GMM Predictions",comparison_key],add_outline=True, size = 100, palette="coolwarm")
    return score,ari,ami,fow
  else:
    sc.pl.pca(adata, color=["GMM Predictions"],add_outline=True, size = 100, palette="coolwarm")
    sc.pl.umap(adata, color=["GMM Predictions"],add_outline=True, size = 100, palette="coolwarm")
    sc.pl.tsne(adata, color=["GMM Predictions"],add_outline=True, size = 100, palette="coolwarm")
    return score

def nonnegativefactorization(data,adata,ground_labels,init,n_clusters,comparison_key = "Ground labels"):
  model = NMF(n_components=n_clusters, init=init, random_state=0)
  pre_labels = model.fit_transform(data)
  labels = np.argmax(pre_labels,axis=1)
  score = silhouette_score(data,labels)
  ari = adjusted_rand_score(ground_labels, labels)
  ami = adjusted_mutual_info_score(ground_labels, labels)
  fow = fowlkes_mallows_score(ground_labels,labels)
  print(f"The silhouete score in the optimal PCA space is: {score}")
  print(f"The adjusted random index (ARI) score with the original labels is: {ari}")
  print(f"The adjusted mutual information (AMI) score with to the original labels is: {ami}")
  print(f"The fowlkes mallows score with the original labels is: {fow}")
  adata.obs['NMF Predictions'] = labels.astype(str)

  sc.pl.pca(adata, color=["NMF Predictions",comparison_key],add_outline=True, size = 100, palette="coolwarm")
  sc.pl.umap(adata, color=["NMF Predictions",comparison_key],add_outline=True, size = 100, palette="coolwarm")
  sc.pl.tsne(adata, color=["NMF Predictions",comparison_key],add_outline=True, size = 100, palette="coolwarm")
  
  return score,ari,ami,fow
  