
test_that("sql variant can be created", {
  expect_s3_class(dbplyr::sql_translation(simulate_datafusion()), "sql_variant")
})
