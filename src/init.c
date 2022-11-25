#define R_NO_REMAP
#define STRICT_R_HEADERS
#include <Rinternals.h>

#include "rustlib/api.h"

SEXP datafusion_testerino_wrapper(void) {
  return Rf_ScalarInteger(datafusion_testerino());
}

static const R_CallMethodDef CallEntries[] = {
  {"datafusion_testerino_wrapper", (DL_FUNC) &datafusion_testerino_wrapper, 0},
  {NULL, NULL, 0}
};

void R_init_datafusion(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
