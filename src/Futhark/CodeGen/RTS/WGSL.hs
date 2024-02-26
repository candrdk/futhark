{-# LANGUAGE TemplateHaskell #-}

-- | Code snippets used by the WebGPU backend as part of WGSL shaders.
module Futhark.CodeGen.RTS.WGSL
  ( arith64
  )
where

import Data.FileEmbed
import Data.Text qualified as T

-- | @rts/wgsl/arith64.wgsl@
arith64 :: T.Text
arith64 = $(embedStringFile "rts/wgsl/arith64.wgsl")
{-# NOINLINE arith64 #-}
