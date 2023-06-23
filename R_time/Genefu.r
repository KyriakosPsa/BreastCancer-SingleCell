# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("genefu")

library(genefu)

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

data(mod1)
scmod1.expos <- subtype.cluster(module.ESR1=mod1$ESR1, module.ERBB2=mod1$ERBB2,
                                module.AURKA=mod1$AURKA, data=expr_matrix, annot=annot_final, do.mapping=TRUE,
                                do.scale=TRUE, plot=FALSE, verbose=TRUE)
results = data.frame(scmod1.expos[["subtype.proba"]])
write.csv(results, file = "scmod1.csv", row.names = TRUE)



