
test_that("sql variant can be created", {
  expect_s3_class(dbplyr::sql_translation(simulate_datafusion()), "sql_variant")
})

test_that("dbplyr::lazy_frame() can generate some SQL from an expression", {
  lf <- dbplyr::lazy_frame(a = TRUE, b = 1, c = 2, d = "z", con = simulate_datafusion())
  lf_agg <- dplyr::summarise(lf, x = sd(b, na.rm = TRUE))
  expect_match(as.character(dbplyr::sql_render(lf_agg)), "^SELECT STDDEV_SAMP")
})
