#define R_NO_REMAP
#define STRICT_R_HEADERS
#include <Rinternals.h>

SEXP dfr_session_context_new(void);
SEXP dfr_session_context_free(SEXP session_xptr);

static const R_CallMethodDef CallEntries[] = {
  {"dfr_session_context_new", (DL_FUNC) &dfr_session_context_new, 0},
  {"dfr_session_context_free", (DL_FUNC) &dfr_session_context_free, 1},
  {NULL, NULL, 0}
};

void R_init_datafusion(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
