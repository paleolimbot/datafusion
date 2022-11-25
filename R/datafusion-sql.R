
#' Datafusion SQL Translation
#'
#' @return
#'   - `simulate_datafusion()` returns an object of class "DatafusionConnection"
#'     that can be used to test SQL generation
#'   - `sql_translation.DatafusionConnection()` returns a [dbplyr::sql_variant()]
#'     that can be used to translate R expressions into SQL.
#' @export
#'
#' @examples
#' simulate_datafusion()
#'
#'
simulate_datafusion <- function() {
  dbplyr::simulate_dbi("DatafusionConnection")
}

#' @rdname simulate_datafusion
#' @importFrom dbplyr sql_translation
#' @export
sql_translation.DatafusionConnection <- function(con) {
  base_postgres <- unclass(dbplyr::sql_translation(dbplyr::simulate_postgres()))
  dbplyr::sql_variant(
    scalar = dbplyr::sql_translator(.parent = base_postgres$scalar),
    aggregate = dbplyr::sql_translator(.parent = base_postgres$aggregate),
    window = dbplyr::sql_translator(.parent = base_postgres$window)
  )
}
