#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @useDynLib datafusion, .registration = TRUE
## usethis namespace: end
NULL

.onLoad <- function(...) {
  vctrs::s3_register("dbplyr::sql_translation", "DatafusionConnection")
}
