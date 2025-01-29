module Futhark.Analysis.Proofs.Rewrite (rewrite, rewriteWithoutRules) where

import Control.Monad (filterM, (<=<))
import Data.List.NonEmpty qualified as NE
import Futhark.Analysis.Proofs.AlgebraBridge (addRelIterator, algebraContext, assume, isFalse, simplify, ($>))
import Futhark.Analysis.Proofs.IndexFn (Cases (..), Domain (..), IndexFn (..), Iterator (..), cases, casesToList)
import Futhark.Analysis.Proofs.IndexFnPlus (intervalEnd)
import Futhark.Analysis.Proofs.Monad (IndexFnM, rollbackAlgEnv)
import Futhark.Analysis.Proofs.Query (isUnknown, isYes)
import Futhark.Analysis.Proofs.Rule (applyRuleBook, rulesIndexFn)
import Futhark.Analysis.Proofs.Symbol (Symbol (..))
import Futhark.Analysis.Proofs.Unify (Renameable, renameSame)
import Futhark.SoP.SoP (SoP, int2SoP, justConstant, sym2SoP, (.*.), (.+.))
import Futhark.SoP.SoP qualified as SoP

normalizeIndexFn :: IndexFn -> IndexFnM IndexFn
normalizeIndexFn = allCasesAreConstants

allCasesAreConstants :: IndexFn -> IndexFnM IndexFn
allCasesAreConstants v@(IndexFn _ (Cases ((Bool True, _) NE.:| []))) = pure v
allCasesAreConstants (IndexFn it (Cases cs))
  | cs' <- NE.toList cs,
    Just vs <- mapM (justConstant . snd) cs' = do
      let ps = map fst cs'
      let sumOfBools =
            SoP.normalize . foldl1 (.+.) $
              zipWith (\p x -> sym2SoP p .*. int2SoP x) ps vs
      -- tell ["Using simplification rule: integer-valued cases"]
      pure $ IndexFn it $ Cases (NE.singleton (Bool True, sumOfBools))
allCasesAreConstants v = pure v

class (Monad m) => Rewritable v m where
  rewrite :: v -> m v

instance Rewritable (SoP Symbol) IndexFnM where
  rewrite = convergeRename simplify

instance Rewritable Symbol IndexFnM where
  rewrite = convergeRename simplify

instance Rewritable IndexFn IndexFnM where
  rewrite = convergeRename (rewrite_ <=< applyRuleBook rulesIndexFn)

convergeRename :: (Renameable t, Eq t) => (t -> IndexFnM t) -> t -> IndexFnM t
convergeRename f x = do
  y <- f x
  (x', y') <- renameSame x y
  if x' == y'
    then pure x'
    else do
      convergeRename f y'

rewrite_ :: IndexFn -> IndexFnM IndexFn
rewrite_ fn@(IndexFn it xs) = normalizeIndexFn =<< simplifyIndexFn
  where
    simplifyIndexFn = algebraContext fn $ do
      addRelIterator it
      ys <- simplifyCases xs
      pure $ IndexFn it ys

    simplifyCases cs = do
      let (ps, vs) = unzip $ casesToList cs
      ps_simplified <- mapM rewrite ps
      cs' <-
        eliminateFalsifiableCases (zip ps_simplified vs)
          >>= eliminateUnreachableCases
          >>= mapM simplifyCase
          >>= mergeCases
      pure $ cases cs'

    -- Simplify x under the assumption that p is true.
    simplifyCase (p, x) | Just _ <- justConstant x =
      pure (p, x)
    simplifyCase (p, x) = rollbackAlgEnv $ do
      -- Take care to convert x first to hopefully get sums of predicates
      -- translated first.
      assume p
      y <- rewrite x
      pure (p, y)

    -- Attempt to merge cases that are equivalent given their predicates.
    -- For example, in
    --   | k > 0  => sum_{j=0}^{k-1} e_j
    --   | k <= 0 => 0
    -- the second case is covered by the first when k <= 0:
    --   | True  => sum_{j=0}^{k-1} e_j
    --
    -- NOTE only tries to merge adjacent cases.
    mergeCases = merge []
      where
        merge acc [] = pure $ reverse acc
        merge [] (a : as) = merge [a] as
        merge ((c, e) : as) ((c', e') : bs) = do
          (_, e_assuming_c') <- simplifyCase (c', e)
          if e_assuming_c' == e'
            then do
              c'' <- rewrite $ c :|| c'
              merge ((c'', e) : as) bs
            else do
              (_, e'_assuming_c) <- simplifyCase (c, e')
              if e'_assuming_c == e
                then do
                  c'' <- rewrite $ c :|| c'
                  merge ((c'', e') : as) bs
                else merge ((c', e') : (c, e) : as) bs

    -- Remove cases for which the predicate can be shown False.
    eliminateFalsifiableCases = filterM (fmap isUnknown . isFalse . fst)

    -- Remove cases for which the predicate causes the current interval to be
    -- empty; hence unreachable.
    eliminateUnreachableCases cs = do
      case it of
        (Forall _ dom@(Cat _ _ b)) -> do
          (unreachables, reachables) <-
            partitionM
              ( \(p, _) -> rollbackAlgEnv $ do
                  assume p
                  isYes <$> (b $> intervalEnd dom)
              )
              cs
          -- Make sure that cases predicates still partition the domain (disjunction is a tautology).
          let q = foldl (:||) (Bool False) $ map fst unreachables
          mapM (\(p, v) -> (,v) <$> rewrite (p :|| q)) reachables
        _ -> pure cs

rewriteWithoutRules :: IndexFn -> IndexFnM IndexFn
rewriteWithoutRules =
  convergeRename rewrite_

-- https://hackage.haskell.org/package/extra-1.8/docs/src/Control.Monad.Extra.html#partitionM
partitionM :: (Monad m) => (a -> m Bool) -> [a] -> m ([a], [a])
partitionM _ [] = pure ([], [])
partitionM f (x : xs) = do
  res <- f x
  (as, bs) <- partitionM f xs
  pure ([x | res] ++ as, [x | not res] ++ bs)
