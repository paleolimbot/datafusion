
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

This requires a Rust compiler, which will use `cargo` to build the
DataFusion Rust library. This won’t work on Windows (not because there’s
anything wrong with Rust, but because something about Rust and msys2
results in too many symbols and the linker can’t deal with it).

## Example

Step one: implement Postgres-flavoured SQL generation so that we can
send it to DataFusion:

``` r
library(datafusion)
library(dplyr)
library(dbplyr)

lazy_frame(a = double(), b = double(), con = simulate_datafusion(), .name = "some_table") |> 
  filter(b > 5) |> 
  summarise(x = sd(a, na.rm = TRUE)) |> 
  sql_render()
#> <SQL> SELECT STDDEV_SAMP(`a`) AS `x`
#> FROM `some_table`
#> WHERE (`b` > 5.0)
```

Step two: build the DataFusion crate and figure out how to pass it SQL.
So far I only have the mechanics to call a simple test function that
returns an integer. Ideally this would be SQL in and ArrowArrayStream
out!

``` r
library(datafusion)

# Just tests a call into rust
datafusion:::testerino()
#> [1] 3
```
