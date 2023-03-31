#define R_NO_REMAP
#define STRICT_R_HEADERS
#include <R.h>
#include <Rinternals.h>

extern "C" {
#include "datafusion.h"
}

template <typename T>
static inline const char* dfr_xptr_class();

template <>
inline const char* dfr_xptr_class<DFSessionContext>() {
  return "df_session_context";
}

template <>
inline const char* dfr_xptr_class<DFDataFrame>() {
  return "df_data_frame";
}

template <>
inline const char* dfr_xptr_class<DFCSVReadOptions>() {
  return "df_csv_read_options";
}

template <>
inline const char* dfr_xptr_class<DFParquetReadOptions>() {
  return "df_parquet_read_options";
}

template <>
inline const char* dfr_xptr_class<DFError>() {
  return "df_error";
}

inline void dfr_ptr_free(DFSessionContext* ptr) {
  df_session_context_free(ptr);
}

inline void dfr_ptr_free(DFDataFrame* ptr) {
  df_data_frame_free(ptr);
}

inline void dfr_ptr_free(DFCSVReadOptions* ptr) {
  df_csv_read_options_free(ptr);
}

inline void dfr_ptr_free(DFParquetReadOptions* ptr) {
  df_parquet_read_options_free(ptr);
}

inline void dfr_ptr_free(DFError* ptr) {
  df_error_free(ptr);
}

template <typename T>
static inline T* dfr_from_xptr(SEXP xptr) {
  if (!Rf_inherits(xptr, dfr_xptr_class<T>())) {
    Rf_error("Expected external pointer with class '%s'", dfr_xptr_class<T>());
  }

  T* ptr = reinterpret_cast<T*>(R_ExternalPtrAddr(xptr));
  if (ptr == nullptr) {
    Rf_error("Can't convert external pointer to NULL to T*");
  }
  return ptr;
}

template <typename T>
static void dfr_finalize_xptr(SEXP xptr) {
  T* ptr = reinterpret_cast<T*>(R_ExternalPtrAddr(xptr));
  if (ptr != NULL) {
    dfr_ptr_free(ptr);
    R_SetExternalPtrAddr(xptr, nullptr);
  }
}

template <typename T>
static inline SEXP dfr_wrap_xptr(T* ptr, SEXP parent) {
  SEXP xptr = PROTECT(R_MakeExternalPtr(ptr, R_NilValue, parent));
  R_RegisterCFinalizer(xptr, &dfr_finalize_xptr<T>);

  SEXP cls = PROTECT(Rf_mkString(dfr_xptr_class<T>()));
  Rf_setAttrib(xptr, R_ClassSymbol, cls);
  UNPROTECT(2);
  return xptr;
}


static inline const char* dfr_as_const_char(SEXP sexp) {
  if (TYPEOF(sexp) != STRSXP || Rf_length(sexp) != 1) {
    Rf_error("Expected character(1) for conversion to const char*");
  }

  SEXP item = STRING_ELT(sexp, 0);
  if (item == NA_STRING) {
    Rf_error("Can't convert NA_character_ to const char*");
  }

  return Rf_translateCharUTF8(item);
}

static inline int dfr_as_int(SEXP sexp) {
  if (Rf_length(sexp) == 1) {
    switch (TYPEOF(sexp)) {
      case REALSXP:
        return REAL(sexp)[0];
      case INTSXP:
        return INTEGER(sexp)[0];
    }
  }

  Rf_error("Expected integer(1) or double(1) for conversion to int");
}

extern "C" SEXP dfr_session_context_new(void) {
  return dfr_wrap_xptr<DFSessionContext>(df_session_context_new(), R_NilValue);
}

extern "C" SEXP dfr_session_context_free(SEXP xptr) {
  DFSessionContext* context = dfr_from_xptr<DFSessionContext>(xptr);
  df_session_context_free(context);
  R_SetExternalPtrAddr(xptr, nullptr);
  return R_NilValue;
}
