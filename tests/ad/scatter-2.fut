-- Simple scatter
-- ==

-- compiled input { [1.0f32, 2.0f32, 3.0f32, 4.0f32] [5.0f32, 4.0f32, 3.0f32, 2.0f32] } output { [79.0f32, 37.0f32, 22.0f32, 12.0f32] }

let scatteromap [n] (is: [n]i64) (vs: [n]f32, xs: [n]f32) : [n]f32 =
  let xs' = map3 (\i v x -> if i < 0 || i > n-1 then v*x else x) is vs xs
  let vs' = map2 (*) vs xs'
  let ys  = scatter xs' is vs'
  let zs  = map2 (*) ys vs
  in  ys 

entry main [n] (is: [n]i64) (vs: [n]f32) (xs: [n]f32) (ys_bar: [n]f32) =
  vjp (scatteromap is) (vs,xs) (copy ys_bar)