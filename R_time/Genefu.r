# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("genefu")
# install.packages("pamr")


library(genefu)
library(pamr)

rm(list = ls())

expr_data = read.csv("gene_expression.csv", header=FALSE)
genes = read.csv("gene_names.csv", header=TRUE)
colnames(expr_data) = genes$GeneID
cells = read.csv("cell_names.csv", header=TRUE)
rownames(expr_data) = cells$CellID
expr_matrix = data.matrix(expr_data)

annot = read.csv("Entrez_id.csv", header=TRUE)
annot_final <- data.frame(Entrez_ID = character(0))

for (gene in genes$GeneID) {
  entrez_id <- annot[annot$GeneID == gene, "Entrez_ID"]
  annot_final <- rbind(annot_final, data.frame(Entrez_ID = entrez_id))
}
colnames(annot_final) = 'EntrezGene.ID'
annot_final = data.matrix(annot_final)


data(scmod1.robust)
data(sig.genius)
# Compute GENIUS risk scores
genius_result <- genius(data = expr_matrix, annot = annot_final, do.mapping = TRUE, do.scale = FALSE)

# Access the risk scores for different subtypes
risk_score_ER_HER2_neg <- genius_result$GENIUSM1  # ER-/HER2- subtype
risk_score_HER2_pos <- genius_result$GENIUSM2     # HER2+ subtype
risk_score_ER_HER2_neg <- genius_result$GENIUSM3  # ER+/HER2- subtype

# Print the risk scores
cat("ER-/HER2- Risk Score:", risk_score_ER_HER2_neg, "\n")
cat("HER2+ Risk Score:", risk_score_HER2_pos, "\n")
cat("ER+/HER2- Risk Score:", risk_score_ER_HER2_neg, "\n")

genius_df <- data.frame(ER_HER2_neg = genius_result$GENIUSM1,
                        HER2_pos = genius_result$GENIUSM2,
                        er_pos_HER2_neg = genius_result$GENIUSM3,
                        score = genius_result$score)

# Export the data frame to a CSV file
write.csv(genius_df, file = "genius_results.csv", row.names = FALSE)



