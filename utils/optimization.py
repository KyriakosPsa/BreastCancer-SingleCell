import optuna
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score , calinski_harabasz_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimizeSpectral(X, k_range: list, neighbors_range :  list, affinity, eval_metric, direction, n_trials=100,seed = 42,show_progress_bar = True):
    """
    Optimizes the parameters for spectral clustering using Optuna.
        X (matrix): The input data matrix.
        k_range (list): A list containing two values indicating the minimum and maximum number of clusters to try.
        neighbors_range (list): A list containing two values indicating the minimum and maximum number of neighbors to try.
        affinity (str): The affinity parameter for spectral clustering. Supported values: "laplacian", "rbf", "nearest_neighbors".
        eval_metric (str): The evaluation metric to use. Supported values: "silhouette_score", "davies_bouldin", "calinski_harabasz".
        direction (str): The direction of optimization. Supported values: "minimize", "maximize".
        n_trials (int, optional): The number of optimization trials to perform. Defaults to 100.
    Returns:s
    best_params (array): The best parameters
    best_scores (array): the best values
    study (object): the study object
        """
    def Spectral_trial(trial):
        # Define the parameter search space
        if (len(k_range) == 2) and (len(neighbors_range) == 2):

          if (affinity == "rbf"):
            n_clusters = trial.suggest_int("n_clusters", k_range[0], k_range[1])
            gamma = trial.suggest_float("gamma", 0.01, 1.0)
            spectral = SpectralClustering(n_clusters=n_clusters,
                                          affinity=affinity,
                                          gamma=gamma,
                                          assign_labels="cluster_qr",
                                          random_state=seed)
            
          elif(affinity == "nearest_neighbors"):
            n_clusters = trial.suggest_int("n_clusters", k_range[0], k_range[1])
            n_neighbors = trial.suggest_int("n_neighbors",neighbors_range[0], neighbors_range[1])
            spectral = SpectralClustering(n_clusters=n_clusters,
                                          affinity=affinity,
                                          n_neighbors=n_neighbors,
                                          assign_labels="cluster_qr",
                                          random_state=seed)
          else: 
            raise ValueError("Unknown affinity parameter, try laplacian, rbf, nearest_neighbors")
        else:
          raise ValueError("k_range, neighbors_range need two items, indicating the min and max clusters, neighbors to try")

        # Find the cluster labels
        labels = spectral.fit_predict(X)
        # Begin evaluation based on the selected metric
        if eval_metric == "silhouette_score":
          try:
              return silhouette_score(X, labels, random_state=seed)
          except ValueError:
              # Return the worst score if it does not converge
              return -1
        elif eval_metric == "davies_bouldin":
          try:
              return davies_bouldin_score(X, labels)
          except ValueError:
              # Return the worst score if it does not converge
              return np.inf
        elif eval_metric == "calinski_harabasz":
            return calinski_harabasz_score(X, labels)
    # Trial pruner
    pruner = optuna.pruners.HyperbandPruner()
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study.optimize(Spectral_trial, n_trials=n_trials, show_progress_bar=show_progress_bar)

    # Get the best parameters and objective value
    best_params = study.best_params
    best_value = study.best_value

    return best_params, best_value


def optimizeGMM(X,metric = "BIC",max_num_components = 10,precomputed_means=None,n_trials=100,seed = 42,precomputed = False,show_progress_bar = False,max_iter = 100):
    """
    Optimize the parameters of Gaussian Mixture Models (GMM) using the specified metric.
    Parameters:
    
        X (array-like): The input data matrix.
        - metric (str, optional): The optimization metric to use. Supported values: "BIC", "AIC". Defaults to "BIC".
        - max_num_components (int, optional): The maximum number of components to try. Defaults to 10. Ignored if precomputed_means is provided.
        - precomputed_means (array-like, optional): Precomputed means for initializing the Gaussian Mixture Model. Defaults to None.
        - n_trials (int, optional): The number of optimization trials to perform. Defaults to 100.
        - seed (int, optional): The random seed for reproducibility. Defaults to 42.
        - precomputed (bool, optional): Whether precomputed means are provided. Defaults to False.
        - show_progress_bar (bool, optional): Whether to show a progress bar during optimization. Defaults to False.
        - max_iter (int, optional): The maximum number of iterations for GMM fitting. Defaults to 100.

    Returns:
      if precomputed == False:
        best_params (array-like): The best parameters found during optimization.
        best_value (float): The best objective value (metric score) found during optimization.
      if precomputed == True:
        params (list): The best parameters for each number of components, 2 to max_num_components
        values (list): The best values for each number of components, 2 to max_num_components
    """
    params = []
    values = []
    def bicTrial(trial,max_num_components = max_num_components):
      if precomputed:
        covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])
        random_state = trial.suggest_int("random_state",1,n_trials)
        gmm = GaussianMixture(n_components = precomputed_means.shape[0],means_init=precomputed_means, covariance_type=covariance_type,random_state = random_state, max_iter=max_iter)
      else:
        covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])
        random_state = trial.suggest_int("random_state",1,n_trials)
        init_params = trial.suggest_categorical("init_params",["kmeans","k-means++"])
        gmm = GaussianMixture(n_components=max_num_components, covariance_type=covariance_type,init_params=init_params,random_state = random_state,max_iter=max_iter)
      # Fit Gaussian Mixture Model
      try:
        gmm.fit(X)
      except ValueError:
        return np.inf
      if metric == "BIC":
        bic_score = gmm.bic(X)
        return bic_score
      elif metric == "AIC":
        aic_score = gmm.aic(X)
        return aic_score
      else:
          raise ValueError ("Invalid metric, choose between BIC, AIC")
    # Define the Optuna study and optimize the objective
    # if means are precomputed just optimize the BIC
    if precomputed:
      pruner = optuna.pruners.HyperbandPruner()
      sampler = optuna.samplers.TPESampler(seed=seed,multivariate=True)
      study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
      study.optimize(bicTrial, n_trials=n_trials, show_progress_bar=show_progress_bar)
      best_params = study.best_params
      return best_params
    # otherwise iterate over the number of components and pick the model which 
    else:   
      for components in range(1, max_num_components + 1):
        pruner = optuna.pruners.HyperbandPruner()
        sampler = optuna.samplers.TPESampler(seed=seed,multivariate=True)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        study.optimize(lambda trial: bicTrial(trial, components), n_trials=n_trials, show_progress_bar=show_progress_bar)
      # Get the best parameters and objective value
        best_params = study.best_params
        best_value = study.best_value
        params.append(best_params)
        values.append(best_value)
      # identify the knee point in the component-BIC score grap
      return params, values
    