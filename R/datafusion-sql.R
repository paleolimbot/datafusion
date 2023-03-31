
#' Datafusion SQL Translation
#'
#' @return
#'   - `simulate_datafusion()` returns an object of class "DatafusionConnection"
#'     that can be used to test SQL generation
#' @export
#'
#' @examples
#' library(dplyr, warn.conflicts = FALSE)
#'
#' lf <- dbplyr::lazy_frame(a = TRUE, b = 1, c = 2, d = "z", con = simulate_datafusion())
#' lf %>% summarise(x = sd(b, na.rm = TRUE))
#' lf %>% summarise(y = cor(b, c), z = cov(b, c))
#'
simulate_datafusion <- function() {
  dbplyr::simulate_dbi("DatafusionConnection")
}

sql_translation.DatafusionConnection <- function(con) {
  base_postgres <- unclass(dbplyr::sql_translation(dbplyr::simulate_postgres()))
  dbplyr::sql_variant(
    scalar = dbplyr::sql_translator(.parent = base_postgres$scalar),
    aggregate = dbplyr::sql_translator(.parent = base_postgres$aggregate),
    window = dbplyr::sql_translator(.parent = base_postgres$window)
  )
}

dummy_use_dplyr <- function() {
  # CMD check complains that dplyr isn't used in the package
  dplyr::n()
}
