import json
import os
import scanpy as sc


def buildGeneset(directory,adata):
  json_files = [file for file in os.listdir(directory) if file.endswith('.json')]
  gene_sets = {}

  for file in json_files:
      file_name = os.path.splitext(file)[0]  # Remove the file extension
      gene_set_name = file_name.split('.')[0]  # Extract the desired part of the file name
      with open(os.path.join(directory, file), 'r') as f:
          gene_set_data = json.load(f)
          # Extract the nested gene set using the appropriate key
          try:
            gene_set = gene_set_data['genes']
          except KeyError:
            gene_set = gene_set_data[gene_set_name]['geneSymbols']
          gene_set_final = set(adata.var_names) & set(gene_set)
          print(f"Genes removed from the gene set: {len(set(gene_set) - gene_set_final)} out of {len(gene_set)}\n")
          gene_sets[gene_set_name] = gene_set_final
  return gene_sets