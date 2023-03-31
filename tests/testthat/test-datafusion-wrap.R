
test_that("df_session_context() works", {
  session <- df_session_context()
  expect_s3_class(session, "df_session_context")
  df_session_context_free(session)
  expect_error(
    df_session_context_free(session),
    "Can't convert external pointer to NULL"
  )
})
