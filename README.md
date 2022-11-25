
<!-- README.md is generated from README.Rmd. Please edit that file -->

# datafusion

<!-- badges: start -->

[![R-CMD-check](https://github.com/paleolimbot/datafusion/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/paleolimbot/datafusion/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

The goal of datafusion is to figure out if an R wrapper around
[DataFusion](https://arrow.apache.org/datafusion/index.html) could ever
be a thing.

## Installation

You can install the development version of datafusion from
[GitHub](https://github.com/) with:

``` r
# install.packages("remotes")
remotes::install_github("paleolimbot/datafusion")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(datafusion)

# Just tests a call into rust
datafusion:::testerino()
#> [1] 3
```
