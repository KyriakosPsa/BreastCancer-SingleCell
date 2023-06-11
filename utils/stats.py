import numpy as np

def medianAbsdev(qc_metric):
  """Median abosulute deviation equation"""
  return np.median(np.abs(qc_metric - np.median(qc_metric)))

def is_outlier(adata, metric: str, nmads: int):
  """Function to identify outliers in QC metric distributions based on medianAbsdev"""
  M = adata.obs[metric]
  outlier = (M < np.median(M) - nmads * medianAbsdev(M)) | (
      np.median(M) + nmads * medianAbsdev(M) < M
  )
  return outlier