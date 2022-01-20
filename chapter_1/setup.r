# PREREQS
# pacman -S gcc-fortran extras/libgit2

# Set up: install  RStan
install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)

# install rethinking

install.packages("remotes")
remotes::install_github("stan-dev/cmdstanr")

install.packages(c("coda","mvtnorm","devtools","dagitty"))
library(devtools)
devtools::install_github("rmcelreath/rethinking")

