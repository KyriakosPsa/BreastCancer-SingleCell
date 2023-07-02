# **Project Description**

In this study, we have successfully replicated and built upon the analysis conducted in a highly influential paper in the field of breast cancer single-cell RNA sequencing (scRNA-seq), titled "Single-cell RNA-seq enables comprehensive tumour and immune cell profiling in primary breast cancer" that can be found with this [link](https://www.nature.com/articles/ncomms15081). Our research not only validates the feasibility of distinguishing between tumor and immune cells at the single-cell level but also sheds light on their characterization using targeted gene sets and unsupervised machine-learning techniques. These findings hold significant promise for advancing personalized medicine strategies in breast cancer. To enhance our analysis, we developed a preprocessing pipeline tailored to our specific research goals. By employing distinct and well-evaluated clustering approaches at each stage of cell separation, we aimed to optimize the accuracy and reliability of our results. Moreover, we incorporated advanced visualization techniques such as UMAP and t-SNE. These additions provided invaluable insights into the underlying organization and relationships within the data, enriching our overall analysis.

**A comprehensive report of our findings and analysis can be found in the `Project_Report.pdf` file.**

# **Data Availability**

Both the publicly available datasets from the reference paper and our own data are available in this [Google Drive Folder](https://drive.google.com/drive/folders/1goDwt_HCBL1fEAhduti1fv9DkAY_GPCE)

# **Instructions**

If you would like to run the code yourself, please follow the instructions below.

## _Library dependencies_

- kneed
- matplotlib
- numpy
- pandas
- scanpy
- anndata
- seaborn
- sklearn
- optuna

## _Creating the workspace_

1. Clone the repository
2. Download the reference paper data from the [Google Drive Folder](https://drive.google.com/drive/folders/1goDwt_HCBL1fEAhduti1fv9DkAY_GPCE) and place it in a folder named: `datasets`.
3. Run the notebooks in the following order 1. `preprocessing.ipynb` 2. `dimensionalityred.ipynb` 3. `cellseperation.ipynb` 4. `R_genefu_results.ipynb` 5. `cancercellanalysis.ipynb` 6. `immunecellanalysis.ipynb`

---
