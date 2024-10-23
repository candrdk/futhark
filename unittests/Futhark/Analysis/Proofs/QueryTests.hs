{-# OPTIONS_GHC -Wno-name-shadowing #-}
{-# OPTIONS_GHC -Wno-orphans #-}

module Futhark.Analysis.Proofs.QueryTests (tests) where

import Futhark.Analysis.Proofs.IndexFn
import Futhark.Analysis.Proofs.Monad
import Futhark.Analysis.Proofs.Query
import Futhark.Analysis.Proofs.Symbol
import Futhark.MonadFreshNames
import Futhark.SoP.SoP (sym2SoP, (~-~))
import Test.Tasty
import Test.Tasty.HUnit

runTest :: IndexFnM a -> a
runTest test = fst $ runIndexFnM test blankNameSource

tests :: TestTree
tests =
  testGroup
    "Proofs.Query"
    [ testCase "Monotonically increasing" $
        run
          ( \(i, _, _, n, _, _, _) -> do
              let fn =
                    IndexFn
                      { iterator = Forall i (Iota (sVar n)),
                        body =
                          cases [(Bool True, sVar i)]
                      }
              ask (CaseIsMonotonic Inc) fn 0
          )
          @?= Yes,
      testCase "Monotonically decreasing" $
        run
          ( \(i, _, _, n, _, _, _) -> do
              let fn =
                    IndexFn
                      { iterator = Forall i (Iota (sVar n)),
                        body =
                          cases [(Bool True, Var n ~-~ Var i)]
                      }
              ask (CaseIsMonotonic Dec) fn 0
          )
          @?= Yes,
      testCase "Monotonicity unknown 1" $
        run
          ( \(i, _, _, n, x, _, _) -> do
              let fn =
                    IndexFn
                      { iterator = Forall i (Iota (sVar n)),
                        body =
                          cases [(Bool True, sym2SoP $ Idx (Var x) (sVar i))]
                      }
              ask (CaseIsMonotonic Inc) fn 0
          )
          @?= Unknown,
      testCase "Monotonicity unknown 2" $
        run
          ( \(i, _, _, n, x, _, _) -> do
              let fn =
                    IndexFn
                      { iterator = Forall i (Iota (sVar n)),
                        body =
                          cases [(Bool True, Var x ~-~ Var i)]
                      }
              ask (CaseIsMonotonic Inc) fn 0
          )
          @?= Unknown,
      testCase "Monotonic constant" $
        run
          ( \(i, _, _, n, x, _, _) -> do
              let fn =
                    IndexFn
                      { iterator = Forall i (Iota (sVar n)),
                        body =
                          cases [(Bool True, sVar x)]
                      }
              ask (CaseIsMonotonic Inc) fn 0
          )
          @?= Yes
    ]
  where
    -- int = int2SoP
    sVar = sym2SoP . Var
    -- a ~+~ b = sym2SoP a .+. sym2SoP b
    -- a ~-~ b = sym2SoP a .-. sym2SoP b

    varsM =
      (,,,,,,)
        <$> newVName "i"
        <*> newVName "j"
        <*> newVName "k"
        <*> newVName "n"
        <*> newVName "x"
        <*> newVName "y"
        <*> newVName "z"

    run f = runTest (varsM >>= f)

    -- -- Less fragile renaming.
    -- actual @??= expected = do
    --   unless (actual == expected) (assertFailure $ msg actual expected)
    --   where
    --     msg actual expected =
    --       docString $
    --         "expected:" <+> pretty expected <> line <> "but got: " <+> pretty actual

    -- addAlgRange vn x y = do
    --   a <- toAlgebra x
    --   b <- toAlgebra y
    --   addRange (Algebra.Var vn) (mkRange a b)
