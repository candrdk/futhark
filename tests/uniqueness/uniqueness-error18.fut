-- Check that unique components of a return tuple do not alias each
-- other.
-- ==
-- error: unique

let main(n: i64): (*[]i64, *[]i64) =
  let a = iota(n) in
  (a, a)
