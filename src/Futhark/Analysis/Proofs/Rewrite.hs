module Futhark.Analysis.Proofs.Rewrite
where

import Futhark.Analysis.Proofs.Unify
import Futhark.SoP.SoP (SoP, sym2SoP, (.+.), int2SoP, (.-.), sopToList, sopFromList, numTerms)
import Futhark.MonadFreshNames
import Futhark.Analysis.Proofs.Symbol (Symbol(..))
import Control.Monad (foldM, msum, (<=<))
import Futhark.SoP.FourierMotzkin (($<=$))
import Futhark.Analysis.Proofs.IndexFn (IndexFnM)
import Data.List (subsequences, (\\))
import Futhark.Analysis.Proofs.Traversals (ASTMapper(..), astMap)
import Futhark.Analysis.Proofs.Refine (refineSymbol)
import Futhark.SoP.Monad (substEquivs)
import Data.Functor ((<&>))
import Language.Futhark (VName)

data Rule a b m = Rule {
    name :: String,
    from :: a,
    to :: Substitution b -> m a,
    sideCondition :: Substitution b -> m Bool
  }

vacuous :: Monad m => b -> m Bool
vacuous = const (pure True)

int :: Integer -> SoP Symbol
int = int2SoP

sVar :: VName -> SoP Symbol
sVar = sym2SoP . Var

hole :: VName -> SoP Symbol
hole = sym2SoP . Hole

class Monad m => Rewritable u m where
  rewrite :: u -> m u

instance Rewritable (SoP Symbol) IndexFnM where
  rewrite = astMap m <=< substEquivs
    where
      m = ASTMapper
        { mapOnSymbol = rewrite,
          mapOnSoP = \sop -> rulesSoP >>= foldM (flip matchSoP) sop
        }

instance Rewritable Symbol IndexFnM where
  rewrite = astMap m
    where
      m = ASTMapper
        { mapOnSoP = rewrite,
          mapOnSymbol = \x -> do
            rulesSymbol
              >>= foldM (flip matchSymbol) (normalize x)
              >>= refineSymbol . normalize
              <&> normalize
        }

      -- TODO Normalize only normalizes Boolean expressions.
      --      Use a Boolean representation that is normalized by construction.
      normalize :: Symbol -> Symbol
      normalize symbol = case toNNF symbol of
          (Not x) -> toNNF (Not x)
          (x :&& y) ->
            case (x, y) of
              (Bool True, b) -> b                       -- Identity.
              (a, Bool True) -> a
              (Bool False, _) -> Bool False             -- Annihilation.
              (_, Bool False) -> Bool False
              (a, b) | a == b -> a                      -- Idempotence.
              (a, b) | a == toNNF (Not b) -> Bool False -- A contradiction.
              (a, b) -> a :&& b
          (x :|| y) -> do
            case (x, y) of
              (Bool False, b) -> b                      -- Identity.
              (a, Bool False) -> a
              (Bool True, _) -> Bool True               -- Annihilation.
              (_, Bool True) -> Bool True
              (a, b) | a == b -> a                      -- Idempotence.
              (a, b) | a == toNNF (Not b) -> Bool True  -- A tautology.
              (a, b) -> a :|| b
          v -> v

      toNNF :: Symbol -> Symbol
      toNNF (Not (Not x)) = x
      toNNF (Not (Bool True)) = Bool False
      toNNF (Not (Bool False)) = Bool True
      toNNF (Not (x :|| y)) = toNNF (Not x) :&& toNNF (Not y)
      toNNF (Not (x :&& y)) = toNNF (Not x) :|| toNNF (Not y)
      toNNF (Not (x :== y)) = x :/= y
      toNNF (Not (x :< y)) = x :>= y
      toNNF (Not (x :> y)) = x :<= y
      toNNF (Not (x :/= y)) = x :== y
      toNNF (Not (x :>= y)) = x :< y
      toNNF (Not (x :<= y)) = x :> y
      toNNF x = x


match_ :: Unify u v m => Rule u v m -> u -> m (Maybe (Substitution v))
match_ rule x = unify (from rule) x >>= checkSideCondition
  where
    checkSideCondition Nothing = pure Nothing
    checkSideCondition (Just s) = do
      b <- sideCondition rule s
      pure $ if b then Just s else Nothing

-- I use this for debugging.
-- matchTRACE :: (Pretty u, Unify u (SoP u) m, Replaceable u (SoP u), Ord u) => Rule (SoP u) (SoP u) m -> SoP u -> m (Maybe (Substitution (SoP u)))
-- matchLOL rule x = do
--   res <- unify (from rule) x >>= checkSideCondition
--   case res of
--     Just r -> do
--       traceM ("\nmatch_\n  " <> prettyString (from rule) <> "\n  " <> prettyString x)
--       traceM (prettyString r)
--       pure res
--     Nothing -> pure Nothing
--   where
--     checkSideCondition Nothing = pure Nothing
--     checkSideCondition (Just s) = do
--       b <- sideCondition rule s
--       pure $ if b then Just s else Nothing

-- Apply SoP-rule with k terms to all matching k-subterms in a SoP.
-- For example, given rule `x + x => 2x` and SoP `a + b + c + a + b`,
-- it matches `a + a` and `b + b` and returns `2a + 2b + c`.
matchSoP :: ( Replaceable u (SoP u)
            , Unify u (SoP u) m
            , Ord u) => Rule (SoP u) (SoP u) m -> SoP u -> m (SoP u)
matchSoP rule sop
  | numTerms (from rule) <= numTerms sop = do
    let (subterms, contexts) = unzip . combinations $ sopToList sop
    -- Get first valid subterm substitution. Recursively match context.
    subs <- mapM (match_ rule . sopFromList) subterms
    case msum $ zipWith (\x y -> (,y) <$> x) subs contexts of
      Just (s, ctx) -> (.+.) <$> matchSoP rule (sopFromList ctx) <*> to rule s
      Nothing -> pure sop
  | otherwise = pure sop
  where
    -- Get all (k-subterms, remaining subterms).
    k = numTerms (from rule)
    combinations xs = [(s, xs \\ s) | s <- subsequences xs, length s == k]

matchSymbol :: Rule Symbol (SoP Symbol) IndexFnM -> Symbol -> IndexFnM Symbol
matchSymbol rule symbol = do
    s :: Maybe (Substitution (SoP Symbol)) <- case from rule of
      x :&& y -> matchCommutativeRule (:&&) x y
      x :|| y -> matchCommutativeRule (:||) x y
      _ -> match_ rule symbol
    maybe (pure symbol) (to rule) s
    where
      matchCommutativeRule op x y =
        msum <$> mapM (match_ rule) [x `op` y, y `op` x]

rulesSoP :: IndexFnM [Rule (SoP Symbol) (SoP Symbol) IndexFnM]
rulesSoP = do
  i <- newVName "i"
  h1 <- newVName "h"
  h2 <- newVName "h"
  h3 <- newVName "h"
  x1 <- newVName "x"
  y1 <- newVName "y"
  pure
    [ Rule
        { name = "Extend sum lower bound (1)"
        , from = LinComb i (hole h1 .+. int 1) (hole h2) (Hole h3)
                   ~+~ Idx (Hole h3) (hole h1)
        , to = \s -> sub s $ LinComb i (hole h1) (hole h2) (Hole h3)
        , sideCondition = vacuous
        }
    , Rule
        { name = "Extend sum lower bound (2)"
        , from = LinComb i (hole h1) (hole h2) (Hole h3)
                   ~+~ Idx (Hole h3) (hole h1 .-. int 1)
        , to = \s -> sub s $
                  LinComb i (hole h1 .-. int 1) (hole h2) (Hole h3)
        , sideCondition = vacuous
        }
    , Rule
        { name = "Extend sum upper bound (1)"
        , from = LinComb i (hole h1) (hole h2 .-. int 1) (Hole h3)
                   ~+~ Idx (Hole h3) (hole h2)
        , to = \s -> sub s $ LinComb i (hole h1) (hole h2) (Hole h3)
        , sideCondition = vacuous
        }
    , Rule
        { name = "Extend sum upper bound (2)"
        , from = LinComb i (hole h1) (hole h2) (Hole h3)
                   ~+~ Idx (Hole h3) (hole h2 .+. int 1)
        , to = \s -> sub s $
                  LinComb i (hole h1) (hole h2 .+. int 1) (Hole h3)
        , sideCondition = vacuous
        }
    , Rule
        { name = "Merge sum-subtractation"
        , from = LinComb i (hole h1) (hole x1) (Hole h2)
                   ~-~ LinComb i (hole h1) (hole y1) (Hole h2)
        , to = \s ->
           sub s $ LinComb i (hole y1 .+. int 1) (hole x1) (Hole h2)
        , sideCondition = \s -> do
            y' <- sub s (Hole y1)
            x' <- sub s (Hole x1)
            rep s y' $<=$ rep s x'
        }
    , Rule
        { name = "[[¬x]] => 1 - [[x]]"
        , from = sym2SoP $ Indicator (Not (Hole h1))
        , to = \s -> sub s $ int 1 .-. sym2SoP (Indicator (Hole h1))
        , sideCondition = vacuous
        }
    ]
  where
    a ~+~ b = sym2SoP a .+. sym2SoP b
    a ~-~ b = sym2SoP a .-. sym2SoP b

-- TODO can all of these be handled by `normalize`? If so, remove.
rulesSymbol :: IndexFnM [Rule Symbol (SoP Symbol) IndexFnM]
rulesSymbol = do
  pure
    []
    -- [ Rule
    --     { name = ":&& identity"
    --     , from = Bool True :&& Var h1
    --     , to = \s -> pure . sop2Symbol . rep s $ Var h1
    --     }
    -- , Rule
    --     { name = ":&& annihilation"
    --     , from = Bool False :&& Var h1
    --     , to = \_ -> pure $ Bool False
    --     }
    -- , Rule
    --     { name = ":|| identity"
    --     , from = Bool False :|| Var h1
    --     , to = \s -> pure . sop2Symbol . rep s $ Var h1
    --     }
    -- , Rule
    --     { name = ":|| annihilation"
    --     , from = Bool True :|| Var h1
    --     , to = \_ -> pure $ Bool True
    --     }
    -- ]
