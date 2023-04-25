import pandas as pd
df = pd.read_csv(
    "GSE75688_GEO_processed_Breast_Cancer_raw_TPM_matrix.txt", delimiter="\t")
print(df.columns)
