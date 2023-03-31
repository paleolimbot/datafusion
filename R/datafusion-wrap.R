
df_session_context <- function() {
  .Call(dfr_session_context_new)
}

df_session_context_free <- function(session) {
  invisible(.Call(dfr_session_context_free, session))
}

df_session_context_register_record_batches <- function(session, name, batches) {

}

df_session_context_register_csv <- function(session, name, url,
                                            options = df_csv_read_options()) {

}

df_session_context_register_parquet <- function(session, name, url,
                                                options = df_parquet_read_options()) {

}

df_session_context_sql <- function(session, name, sql) {

}

df_session_context_deregister <- function(session,  name) {

}

df_data_frame_show <- function(data_frame) {

}

df_data_frame_export <- function(data_frame) {

}

df_data_frame_free <- function(data_frame) {

}

df_csv_read_options <- function(has_header = NULL, delimter = NULL, schema = NULL,
                                schema_infer_max_records = NULL, file_extension = NULL,
                                table_partition_columns = NULL) {

}

df_parquet_read_options <- function(file_extension = NULL,
                                    table_partition_columns = NULL,
                                    pruning = NULL) {

}

df_error <- function(code, message) {

}

df_error_get_message <- function(error) {

}

df_error_get_code <- function(error) {

}
