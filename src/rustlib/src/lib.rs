
extern crate datafusion;

use datafusion::prelude::*;

#[no_mangle]
pub extern fn datafusion_testerino() -> i32 {
  // Make sure we can call stuff from datafusion
  let ctx = SessionContext::new();

  // Make sure we can return something
  3
}
