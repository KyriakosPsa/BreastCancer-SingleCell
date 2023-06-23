# rforge <- "http://r-forge.r-project.org"
# install.packages("estimate", repos=rforge, dependencies=TRUE)

# help(package="estimate")

##run##
library(estimate)
library(utils)

input_file <- file.path(getwd(), "input_Estimate_format.txt")

filterCommonGenes(input.f = input_file, output.f = "output_file.gct",id="GeneSymbol")

estimateScore("output_file.gct", "estimate_score.gct", platform="affymetrix")

plotPurity(scores="estimate_score.gct", samples="s516", platform="affymetrix")

plotPurity(scores="estimate_score.gct", platform="affymetrix")