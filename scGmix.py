# libraries
import numpy as np
import scanpy as sc
import anndata as adata
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering 
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
import warnings
warnings.filterwarnings("ignore")
import pickle
# Consider aesthetics
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams.update({'font.size': 18})
sc.set_figure_params(scanpy=True, dpi=80, dpi_save=300, 
                    frameon=True, vector_friendly=True, fontsize=18, figsize=(8,8),
                    format='png', ipython_format='png2x')
#File dependancies 
from utils.optimization import optimizeGMM, optimizeSpectral  
from utils.plotting import plot_bic, make_ellipses_joint, posterior_heatmap, plot_state_cellsums, plot_pca

class scgmix():
    """Single Cell Gaussian mixture model pipeline
    
    Class variables:
    -----------------
    - adata (Anndata object): sc-RNAseq expression counts matrix in anndata format
    - method (string): Method used for dimensionality reduction, either PCA, TSEN or UMAP
    - rand_seed (int): A seed for reproducability

    Instance Methods:
    -----------------
    `preprocess`
    `dimreduction`
    `mix`
    `visualize`
    `savefile`
    `savemodel`
    """
    def __init__(self,adata,method ="PCA", rand_seed = 42, model = None):
        self.adata = adata
        self.model = model
        self.method = method
        self.rand_seed = rand_seed
        self.row, self.cols = self.adata.shape

#### Utility ####################################################################################################
    def savefile(self,filenamepath):
      self.adata.write(filenamepath)

    def savemodel(self,filenamepath):
      with open(filenamepath, "wb") as file:
          pickle.dump(self.model, file)

#### STAGE 1: PREPROCESSING INSTANCE METHOD ################################################################################################################
    def preprocess(self, mads_away = 5,feature_selection = False, min_mean=0.0125, max_mean=3, min_disp=0.5):
      """
      Performs preprocessing steps on the data.

      Parameters:
      - mads_away (int): Number of median absolute deviations (MADs) away from the median for cell filtering. Default is 5.
      - feature_selection (bool): Whether to perform feature selection using scanpy's highly variable gene selection. Default is False.
      - min_mean (float): The minimum mean expression threshold for highly variable gene selection. Ignored if feature_selection is False. Default is 0.0125.
      - max_mean (float): The maximum mean expression threshold for highly variable gene selection. Ignored if feature_selection is False. Default is 3.
      - min_disp (float): The minimum dispersion threshold for highly variable gene selection. Ignored if feature_selection is False. Default is 0.5.
      """
      # PRIVATE preprocess sub-methods############################
      def _qcMetrics():
        """Void functin that appends qc data on the adata object inplace
        copy of scanpy's sc.pp.calculate_qc_metrics"""
        sc.pp.calculate_qc_metrics(self.adata,
                                  percent_top = None,
                                  log1p= True,
                                  inplace = True)
      
      def _medianAbsdev(qc_metric):
        """Function that returns the median absolute deviation for a QC metric"""
        return np.median(np.abs(qc_metric - np.median(qc_metric)))
      
      def _filter():
        """Void function that handles cell and gene filtering, cells are removed when their log gene counts, or
        log total expression count are above or below mads_away (absolute deviations away from the median) in both
        mentioned distributions. Gene are removed if they have 0 gene expression for all cells
        """
        m1 = self.adata.obs["log1p_n_genes_by_counts"] # metric 1 for cell filtering
        m2 = self.adata.obs["log1p_total_counts"] # metric 2 for cell filtering 
        m3 = self.adata.var["n_cells_by_counts"] # metric 3 for gene filtering
        # cell filtering
        cell_mask = (m1 < np.median(m1) - mads_away * np.median(m1)) | (m1 > np.median(m1) + mads_away * _medianAbsdev(m1) ) |\
        (m2 < np.median(m2) - mads_away * _medianAbsdev(m2)) | (m2 > np.median(m2) + mads_away)
        self.adata = self.adata[~cell_mask]
        # gene filtering
        gene_mask = self.adata.X.sum(axis=0) == 0
        self.adata = self.adata[:, ~gene_mask]

      def _normalize():
        """Void function the normalizes the counts and log transforms them after adding the value of 1"""
        # Normalization
        sc.pp.normalize_total(self.adata, target_sum=None, inplace=True)
        # log1p transform
        self.adata.X = sc.pp.log1p(self.adata.X)

      # preprocessing method execution 
      #-----------------------------------------
      _qcMetrics() # QC 
      _filter() # QC
      _normalize() # normalization, log transformation
      # optinal higly variable gene selection
      if feature_selection:
        sc.pp.highly_variable_genes(self.adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

#### STAGE 2: DIMENSIONALITY REDUCTION ISNTANCE METHOD ####################################################################################################################
    def dimreduction(self, n_pcs = 100, pc_selection_method = "screeplot",  n_neighbors=15, min_dist = 0.1,
                  use_highly_variable = False,variance_threshold = 90, verbose = True,plot_result = False):
      """
      Performs dimensionality reduction and optimal component selection on the data using the specified method: PCA, TSNE or UMAP 
      & screeplot plot, variance threshold or kaiser's rule.
      Parameters:
      - method (str): The dimensionality reduction method to use. Options are "PCA", "TSNE", or "UMAP". Default is "PCA".
      - n_pcs (int): Number of principal components to use in the initial pca.
      - pc_selection_method (str): The method to determine the optimal number of principal components. Options are "screeplot", "kaiser", or "variance". Default is "screeplot".
      - n_neighbors (int): The number of neighbors to consider for UMAP. Default is 15., ignored for PCA, TSNE
      - min_dist (float): The minimum distance between points for UMAP. Default is 0.1, ignored for PCA, TSNE.
      - use_highly_variable (bool): Whether to use only highly variable genes for PCA. Default is False.
      - variance_threshold (int): The threshold for variance-based principal component selection. Default is 90, ignored if "knee", or "kaiser".
      - verbose (bool): Wether or not to print the optimal number of components found.
      - plot_result (bool):  Wether or not to plot the results of the dimensionality reduction.
      """
      # PRIVATE optimal number of principal components selection methods
      def _kneemethod(explained_variance):
        """Screeplot plot knee method, knee point identified via the kneed library"""
        kneedle = KneeLocator(x =np.arange(1,explained_variance.shape[0]+1,1), y = explained_variance, S=1.0, curve="convex", direction="decreasing")
        optimal_pcs = explained_variance[:round(kneedle.knee)]
        return optimal_pcs
      
      def _variancethreshold(explained_variance,threshold):
        """cummulative variance threshold"""
        cummulative_variance = np.cumsum(explained_variance)
        optimal_pcs = cummulative_variance[cummulative_variance <= threshold]
        return optimal_pcs
      
      def _kaiserule(explained_variance):
        """kaiser's rule threshold"""
        optimal_pcs = explained_variance[explained_variance > 1]
        return optimal_pcs
      
      def _compute():
        """Compute the dimensionality reduction representation and the optimal number of components"""
        # Pca is needed as initilization for TSNE, UMAP even if its not picked as the method the user chooses
        if use_highly_variable:
          # if we use higly variable limit the n_comps of the initial pca to the number of higly variable genes - 1
          sc.pp.pca(self.adata, svd_solver ='arpack',use_highly_variable = use_highly_variable, n_comps = (self.adata.var['highly_variable'] == True).sum() + -1)
        else:
          sc.pp.pca(self.adata, svd_solver ='arpack',use_highly_variable = use_highly_variable, n_comps = n_pcs)
        explained_variance = self.adata.uns['pca']['variance']
        # Check for the selected principal componenet selection method
        if pc_selection_method == "screeplot":
          optimal_pcs = _kneemethod(explained_variance)
        elif pc_selection_method == "kaiser":
          optimal_pcs = _kaiserule(explained_variance)
        elif pc_selection_method == "variance":
          optimal_pcs = _variancethreshold(explained_variance,variance_threshold)
        else:
          raise ValueError("Invalid pc selection method, choose between screeplot, kaiser, variance")
        if verbose:
            print(f"{pc_selection_method} selected {optimal_pcs.shape[0]} principal components out of {n_pcs}")
        # Check the dimensionality reduction method
        if self.method == "PCA":
          # Restrict the componenets to the optimal number identified by `pc_selection_method`
          self.adata.uns['pca']['variance'] = self.adata.uns['pca']['variance'][optimal_pcs.shape[0]]
          self.adata.obsm['X_pca'] = self.adata.obsm['X_pca'][:,:optimal_pcs.shape[0]]
        elif self.method == "TSNE":
          # Run tsne with the suggested parameters from [The art of using t-SNE for single-cell transcriptomics]
          n = self.row/100
          if n/100 > 30:
            perplexity = 30 + n/100 
          else:
            perplexity = 30
          if n/12 > 200:
            learning_rate = n/12
          else:
            learning_rate = 200
          sc.tl.tsne(self.adata, 
                    n_pcs = optimal_pcs.shape[0], perplexity = perplexity, early_exaggeration=12, 
                    learning_rate = learning_rate, random_state = self.rand_seed, use_fast_tsne = False)
        elif self.method  == "UMAP":
          # Parameters for UMAP are up to the user
          sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, random_state= self.rand_seed, n_pcs= optimal_pcs.shape[0])
          sc.tl.umap(self.adata, min_dist= min_dist, random_state= self.rand_seed)
        else:
          raise ValueError("Invalid dimensionality reduction method, choose between PCA, TSNE, UMAP")

      def _plot():
        """plot the results"""
        if self.method == "PCA":
          sc.pl.pca(self.adata,color="total_counts",add_outline = True, size = 100, title= "PCA, cell total expression counts colormap",show=False)
          plt.xlabel(f"PC1: {round(self.adata.uns['pca']['variance_ratio'][0]*100,2)}% of total variance")
          plt.ylabel(f"PC2: {round(self.adata.uns['pca']['variance_ratio'][1]*100,2)}% of total variance")
          plt.show()
        elif self.method == "TSNE":
          sc.pl.tsne(self.adata,add_outline = True, size = 100,title = "t-SNE, cell total expression counts colormap",color="total_counts")
        elif self.method  == "UMAP":
          sc.pl.umap(self.adata, add_outline=True, size=100, title="UMAP, cell total expression counts colormap", color="total_counts")

      # dimensionality reduction method execution 
      #-----------------------------------------
      _compute()
      if plot_result:
        _plot()

### STAGE 3: CLUSTERING ####################################################################################################################
    def mix(self,preclustering_method = "spectral",  enable_preclustering = False, leiden_resolution = 1.0,
            criterion = "BIC" , n_trials = 100, verbose = True, max_iter = 1000,max_num_components = 5, user_means = None,show_progress_bar = True):
      """
      Performs clustering on the data using Gaussian Mixture Models (GMM).
      The fitted model can be accesed via `.model`.

      Parameters:
        - representation (str): The data representation to use for clustering. Options are "PCA", "TSNE", or "UMAP". Default is "PCA".
        - preclustering_method (str): The preclustering method to use for initialization. Options are "spectral" or "leiden". Default is "spectral".
        - enable_preclustering (bool): Whether to compute means for GMM initialization using preclustering. Default is False.
        - leiden_resolution (float): The resolution parameter for Leiden clustering. Ignored if preclustering_method is not "leiden". Default is 1.0.
        - criterion (str): The criterion to optimize for GMM selection. Options are "BIC" or "AIC". Default is "BIC".
        - n_trials (int): The number of trials for GMM optimization. Default is 100.
        - verbose (bool): Whether to print the BIC value of the optimized GMM and plot the BIC/component lineplot. Default is True.
        - max_iter (int): The maximum number of iterations for GMM fitting. Default is 1000, used both for the final fitting and trials.
        - max_num_components (int): The maximum number of components to consider for GMM optimization. Default is 5, ignored if enable_preclustering = True.
        - user_means (ndarray): User-defined means for GMM initialization. Ignored if preclustering_method is not "user". Default is None.
        - show_progress_bar (bool): Whether to show a progress bar during optimization. Default is True.

      Returns:
        - gmm_study (optuna.Study): Optuna study object containing the optimization results if return_study is True.
      """

      # PRIVATE clustering methods########
      def _maxprevnextdiff(list):
        """Returns the elemnt with the largest combined difference with the next and previous element of a list"""
        arr = np.array(list)
        total_diff = np.diff(arr)
        max_diff_index = np.argmin(total_diff)  
        return max_diff_index + 1 # +1 to account for the correct index

      def _computemeans(X,labels):
        """Finds the cluster centers after labels have been identified with preclustering"""
        unique_labels = np.unique(labels)
        cluster_means = [np.mean(X[labels == label], axis=0) for label in unique_labels]
        cluster_means = np.array(cluster_means)  
        return cluster_means
      
      if self.method == "PCA":
        self.X = self.adata.obsm['X_pca']
        reps = 'X_pca'
      elif self.method == "TSNE":
        self.X = self.adata.obsm['X_tsne']
        reps = 'X_tsne'
      elif self.method == "UMAP":
        self.X = self.adata.obsm['X_umap']
        reps = 'X_umap'
      else:
        raise ValueError("Invalid data representation, choose between X_pca, X_tsne, X_umap")
      # Precomputed mixture components means with preclustering 
      if enable_preclustering:
        # Precompute mixture components means with spectral clustering
        if preclustering_method == "spectral":
          best_params, _ = optimizeSpectral(X = self.X, k_range =  [2,30], neighbors_range = [3,40],affinity = "nearest_neighbors",show_progress_bar=show_progress_bar, 
                                                                    eval_metric = "silhouette_score",direction = "maximize", n_trials=50, seed = self.rand_seed)
          sp = SpectralClustering(n_clusters=best_params['n_clusters'],
                                  n_neighbors=best_params['n_neighbors'],
                                  affinity="nearest_neighbors",
                                  assign_labels= 'cluster_qr')
          labels = sp.fit_predict(self.X)
          means = _computemeans(self.X,labels)
        # Precomputed mixture components means with leiden clustering 
        elif preclustering_method == "leiden":
          sc.pp.neighbors(self.adata, use_rep=reps)
          sc.tl.leiden(self.adata, resolution=leiden_resolution, key_added="leiden_"+ reps)
          labels = self.adata.obs["leiden_" + reps]
          means = _computemeans(self.X,labels)
        elif preclustering_method == "user":
          means = user_means
        else:
          raise ValueError("Invalid preclustering method, choose between spectral, leiden or user")
        # Optimized the GMM with the precomputed means
        best_params = optimizeGMM(X = self.X,metric = criterion,precomputed_means=means,n_trials=n_trials, seed=self.rand_seed,max_iter=max_iter,
                                  show_progress_bar =show_progress_bar,precomputed = True)
        # optimize the GMM model
        gmm = GaussianMixture(n_components=means.shape[0], 
                              means_init=means,
                              covariance_type= best_params['covariance_type'],
                              random_state= best_params['random_state'],
                              max_iter=max_iter)
        self.model = gmm.fit(self.X)
        if verbose:
          print(f"Gaussian Mixture model with {criterion} = {gmm.bic(self.X)}")
          print(self.model)
      # Fully automated optimization without precomputed means
      else:
        best_params_list, best_values_list = optimizeGMM(X = self.X,metric = criterion,max_num_components = max_num_components,n_trials=n_trials,seed= self.rand_seed,
                                                        show_progress_bar =show_progress_bar, precomputed = False)
        max_diff_idx = _maxprevnextdiff(best_values_list)
        best_params = best_params_list[max_diff_idx] # find the corresponding best parameters
        components = max_diff_idx + 1 # +1 to account for the fact that components start from 1
        gmm = GaussianMixture(n_components= components,
                              covariance_type= best_params['covariance_type'],
                              random_state= best_params['random_state'],
                              init_params= best_params['init_params'],
                              max_iter=max_iter)
        self.model = gmm.fit(self.X)
        if verbose:
          print(f"Gaussian Mixture model with {criterion} = {self.model.bic(self.X)}")
          print(self.model)
          plot_bic(max_num_components,best_values_list,components,criterion = criterion)

    ### Stage 4 visualization##############################################################
    def visualize(self,membership_threshold=0.90,cmap = "bwr"):
      """
      Visualizes clustering results using Gaussian Mixture Models (GMM).
      Args:
            membership_threshold (float, optional): The threshold for determining 
            cluster membership. Defaults to 0.90.
            cmap (str, optional): The colormap to be used for visualizations. Defaults to "bwr".
      Returns:
            labels: The cluster labels assigned by the GMM model.
            posteriorprob: The posterior probabilities for cell.
            jointprob: Joint probability distribution for each cell
            marginalprob: Marginal probability distribution for each cell
      """
      labels = self.model.predict(self.X)
      self.adata.obs["GMM"] = labels
      posteriorprob = self.model.predict_proba(self.X)
      jointprob = posteriorprob * self.model.weights_
      marginalprob = jointprob.sum(axis = 1)
      make_ellipses_joint(gmm= self.model, X=self.X,labels=labels, proba=posteriorprob,cmap=cmap,title=self.method,joint=False)
      make_ellipses_joint(gmm= self.model, X=self.X,labels=labels, proba=marginalprob,cmap=cmap,title=self.method,joint=True)
      posterior_heatmap(posteriorprob,cmap=cmap)
      posterior_heatmap(jointprob,cmap=cmap,c_label="Joint Probability")
      plot_state_cellsums(posteriorprob,threshold = membership_threshold,labels=labels)
      if self.method != "PCA":
        plot_pca(self.adata,color=labels, title="Clusters infered in the PCA space")
      return labels, posteriorprob, jointprob, marginalprob
