# Set up: install  RStan
install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)

# install rethinking
install.packages(c("coda","mvtnorm","devtools","dagitty"))
library(devtools)
devtools::install_github("rmcelreath/rethinking")

