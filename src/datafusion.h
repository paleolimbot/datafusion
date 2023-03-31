#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * \enum DFErrorCode
 * \brief Error category
 *
 */
typedef enum DFErrorCode {
  DF_ERROR_CODE_ARROW,
  DF_ERROR_CODE_PARQUET,
  DF_ERROR_CODE_AVRO,
  DF_ERROR_CODE_OBJECT_STORE,
  DF_ERROR_CODE_IO,
  DF_ERROR_CODE_SQL,
  DF_ERROR_CODE_NOT_IMPLEMENTED,
  DF_ERROR_CODE_INTERNAL,
  DF_ERROR_CODE_PLAN,
  DF_ERROR_CODE_SCHEMA,
  DF_ERROR_CODE_EXECUTION,
  DF_ERROR_CODE_RESOURCES_EXHAUSTED,
  DF_ERROR_CODE_EXTERNAL,
  DF_ERROR_CODE_JIT,
} DFErrorCode;

typedef struct DFCSVReadOptions DFCSVReadOptions;

/**
 * \struct DFDataFrame
 * \brief A struct for data frame.
 *
 * You get execution result as a data frame.
 *
 * You need to free data frame by `df_data_frame_free()` when no
 * longer needed.
 */
typedef struct DFDataFrame DFDataFrame;

/**
 * \struct DFError
 * \brief A struct that holds error information.
 *
 * You can access to error information by `df_error_get_code()` and
 * `df_error_get_message()`.
 *
 * You need to free error information by `df_error_free()` when no
 * longer needed.
 */
typedef struct DFError DFError;

typedef struct DFParquetReadOptions DFParquetReadOptions;

/**
 * \struct DFSessionContext
 * \brief An entry point of DataFusion API.
 *
 * You need to create `DFSessionContext` to use DataFusion API.
 */
typedef struct DFSessionContext DFSessionContext;

/**
 * \struct DFArrowSchema
 * \brief Same as the `ArrowSchema` struct in the Arrow C data interface
 *
 * See also: https://arrow.apache.org/docs/format/CDataInterface.html#the-arrowschema-structure
 */
typedef struct DFArrowSchema {
  const char *format;
  const char *name;
  const char *metadata;
  int64_t flags;
  int64_t n_children;
  struct DFArrowSchema **children;
  struct DFArrowSchema *dictionary;
  void (*release)(struct DFArrowSchema *schema);
  void *private_data;
} DFArrowSchema;

/**
 * \struct DFArrowArray
 * \brief Same as the `ArrowArray` struct in the Arrow C data interface
 *
 * See also: https://arrow.apache.org/docs/format/CDataInterface.html#the-arrowarray-structure
 */
typedef struct DFArrowArray {
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void **buffers;
  struct DFArrowArray **children;
  struct DFArrowArray *dictionary;
  void (*release)(struct DFArrowArray *array);
  void *private_data;
} DFArrowArray;

struct DFError *df_error_new(enum DFErrorCode code, const char *message);

/**
 * \brief Free the given `DFError`.
 *
 * \param _error A `DFError` returned by `df_*()` functions.
 *
 * # Safety
 *
 * This function should not be called with `error` that is not
 * created by `df_error_new()`.
 *
 * This function should not be called for the same `error` multiple
 * times.
 */
void df_error_free(struct DFError *_error);

/**
 * \brief Get a message of this error.
 *
 * \param error A `DFError`.
 * \return A message of this error.
 *
 * # Safety
 *
 * This function should not be called with `error` that is not
 * created by `df_error_new()`.
 *
 * This function should not be called with `error` that is freed by
 * `df_error_free()`.
 */
const char *df_error_get_message(struct DFError *error);

/**
 * \brief Get a code of this error.
 *
 * \param error A `DFError`.
 * \return A code of this error.
 *
 * # Safety
 *
 * This function should not be called with `error` that is not
 * created by `df_error_new()`.
 *
 * This function should not be called with `error` that is freed by
 * `df_error_free()`.
 */
enum DFErrorCode df_error_get_code(struct DFError *error);

/**
 * \brief Free the given `DFDataFrame`.
 *
 * \param _data_frame A `DFDataFrame`.
 *
 * # Safety
 *
 * This function should not be called for the same `data_frame`
 * multiple times.
 */
void df_data_frame_free(struct DFDataFrame *_data_frame);

/**
 * \brief Show the given data frame contents to the standard output.
 *
 * \param data_frame A `DFDataFrame` to be shown.
 * \param error Return location for a `DFError` or `NULL`.
 */
void df_data_frame_show(struct DFDataFrame *data_frame, struct DFError **error);

int64_t df_data_frame_export(struct DFDataFrame *data_frame,
                             struct DFArrowSchema **c_abi_schema_out,
                             struct DFArrowArray ***c_abi_record_batches_out,
                             struct DFError **error);

/**
 * \brief Create a new `DFSessionContext`.
 *
 * \return A newly created `DFSessionContext`.
 *
 *   It should be freed by `df_session_context_free()` when no longer
 *   needed.
 */
struct DFSessionContext *df_session_context_new(void);

/**
 * \brief Free the given `DFSessionContext`.
 *
 * \param _context A `DFSessionContext` created by
 *   `df_session_context_new()`.
 *
 * # Safety
 *
 * This function should not be called with `context` that is not
 * created by `df_session_context_new()`.
 *
 * This function should not be called for the same `context`
 * multiple times.
 */
void df_session_context_free(struct DFSessionContext *_context);

struct DFDataFrame *df_session_context_sql(struct DFSessionContext *context,
                                           const char *sql,
                                           struct DFError **error);

bool df_session_context_deregister(struct DFSessionContext *context,
                                   const char *name,
                                   struct DFError **error);

bool df_session_context_register_record_batches(struct DFSessionContext *context,
                                                const char *name,
                                                struct DFArrowSchema *c_abi_schema,
                                                struct DFArrowArray **c_abi_record_batches,
                                                size_t n_record_batches,
                                                struct DFError **error);

struct DFCSVReadOptions *df_csv_read_options_new(void);

void df_csv_read_options_free(struct DFCSVReadOptions *_options);

void df_csv_read_options_set_has_header(struct DFCSVReadOptions *options, bool has_header);

bool df_csv_read_options_get_has_header(struct DFCSVReadOptions *options);

void df_csv_read_options_set_delimiter(struct DFCSVReadOptions *options, uint8_t delimiter);

uint8_t df_csv_read_options_get_delimiter(struct DFCSVReadOptions *options);

bool df_csv_read_options_set_schema(struct DFCSVReadOptions *options,
                                    struct DFArrowSchema *schema,
                                    struct DFError **error);

struct DFArrowSchema *df_csv_read_options_get_schema(struct DFCSVReadOptions *options,
                                                     struct DFError **error);

void df_csv_read_options_set_schema_infer_max_records(struct DFCSVReadOptions *options,
                                                      uintptr_t n);

uintptr_t df_csv_read_options_get_schema_infer_max_records(struct DFCSVReadOptions *options);

bool df_csv_read_options_set_file_extension(struct DFCSVReadOptions *options,
                                            const char *file_extension,
                                            struct DFError **error);

char *df_csv_read_options_get_file_extension(struct DFCSVReadOptions *options);

bool df_csv_read_options_set_table_partition_columns(struct DFCSVReadOptions *options,
                                                     const char *const *columns,
                                                     uintptr_t n_columns,
                                                     struct DFError **error);

char **df_csv_read_options_get_table_partition_columns(struct DFCSVReadOptions *options,
                                                       uintptr_t *n_columns);

bool df_session_context_register_csv(struct DFSessionContext *context,
                                     const char *name,
                                     const char *url,
                                     struct DFCSVReadOptions *options,
                                     struct DFError **error);

struct DFParquetReadOptions *df_parquet_read_options_new(void);

void df_parquet_read_options_free(struct DFParquetReadOptions *_options);

bool df_parquet_read_options_set_file_extension(struct DFParquetReadOptions *options,
                                                const char *file_extension,
                                                struct DFError **error);

char *df_parquet_read_options_get_file_extension(struct DFParquetReadOptions *options);

bool df_parquet_read_options_set_table_partition_columns(struct DFParquetReadOptions *options,
                                                         const char *const *columns,
                                                         uintptr_t n_columns,
                                                         struct DFError **error);

char **df_parquet_read_options_get_table_partition_columns(struct DFParquetReadOptions *options,
                                                           uintptr_t *n_columns);

void df_parquet_read_options_set_pruning(struct DFParquetReadOptions *options, bool pruning);

bool df_parquet_read_options_get_pruning(struct DFParquetReadOptions *options);

bool df_session_context_register_parquet(struct DFSessionContext *context,
                                         const char *name,
                                         const char *url,
                                         struct DFParquetReadOptions *options,
                                         struct DFError **error);
