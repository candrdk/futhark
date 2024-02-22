module Futhark.Analysis.View.Rules where

import Futhark.Analysis.View.Representation
import Control.Monad.RWS.Strict hiding (Sum)
import qualified Data.Map as M
import Debug.Trace (trace, traceM)
import Futhark.Util.Pretty (prettyString)
import Data.List.NonEmpty qualified as NE
-- import Control.Monad.Trans.State.Lazy qualified as S
import Control.Monad.State qualified as S
import Control.Monad.Identity

substituteViews :: View -> ViewM View
substituteViews view = do
  knownViews <- gets views
  pure $ idMap (m knownViews) view
  where
    m vs =
      ASTMapper
        { mapOnExp = onExp vs,
          mapOnIf = pure
        }
    onExp _ e@(Var {}) = pure e
    onExp vs e@(Idx (Var xs) i) =
      case M.lookup xs vs of
        -- XXX check that domains are compatible
        -- XXX use index i (for starts, just support simple indexing only?)
        -- XXX merge cases (add cases first, lol)
        Just (Forall j d2 e2) ->
          trace ("🪸 substituting " <> prettyString e <> " for " <> prettyString e2)
          pure e2
        _ -> pure e
    onExp vs v = astMap (m vs) v

-- Hoists case expressions to be the outermost contructor
-- in the view expression by merging cases.
-- 1. seems like a fold where the accumulator is the new Exp
--    that always maintains an outermost Case "invariant"
hoistCases :: View -> ViewM View
-- hoistCases = pure
hoistCases (Forall i dom e) = do
  traceM ("🎭 hoisting ifs")
  let cases = hoistCases' e
  pure $ Forall i dom (Cases $ NE.fromList $ cases)
  -- pure $ Forall i dom (Cases $ NE.fromList $ g cases)
  -- where
  --   g [] = []
  --   g (x:y:xs) = (x,y) : g xs

-- hoistCases' :: [Exp] -> Exp -> [Exp]
-- hoistCases' acc (Var x) = $ Var x
-- hoistCases' acc (Array ts) = pure $ Array ts
-- hoistCases' acc (If c t f) = pure $ If c t f
-- hoistCases' acc (Sum i lb ub e) = pure $ Sum i lb ub e
-- hoistCases' acc (Idx xs i) = pure $ Idx xs i
-- hoistCases' acc (SoP sop) = pure $ SoP sop
-- hoistCases' acc Recurrence = pure Recurrence
-- hoistCases' acc (Bool x) = pure $ Bool x
-- hoistCases' acc (Not x) = pure $ Not x
-- hoistCases' acc (x :== y) = pure $ x :== y
-- hoistCases' acc (x :< y) = pure $ x :< y
-- hoistCases' acc (x :> y) = pure $ x :> y
-- hoistCases' acc (x :&& y) = pure $ x :&& y
-- hoistCases' acc (Cases cases) = pure $ Cases cases

-- foldExp :: (a -> [b] -> b) -> Exp a -> b
-- foldExp f = go where
--     go (Cases cases) = f (map go cases)

-- hoistCases' :: Exp -> [Exp]
-- hoistCases' e = do
--   astMap (m (Bool True)) e
--   where
--     m predicate = ASTMapper { mapOnExp = onExp predicate }
--     -- Want
--     --   onExp pred :: Exp -> m Exp
--     -- where m is the list monad, so
--     --   onExp pred :: Exp -> [Exp]
--     -- Hoping to get an expression tree for every predicate.
--     onExp :: Exp -> Exp -> [Exp]
--     onExp _ (Var x) = pure (Var x)
--     onExp p (If c t f) = do
--       t' <- onExp p t
--       f' <- onExp p f
--       [t', f']
--     -- onExp p (Cases cases) = do
--     --   mconcat $
--     --     mapM (\(p', e') -> astMap (m (p :&& p')) e') (NE.toList cases)
--     --   -- XXX types check if we mconcat here, figure out what we actually
--     --   -- want to do instead. Wait, do we want mconcat?
--     --   -- The problem is that we are discarding the predicates!
--     onExp p v = astMap (m p) v

-- newtype Lol a = Lol (S.StateT [a] Identity a)
-- newtype Lol a = Lol (S.State [(a,a)] a)
-- ^ I don't get why this doesn't work in place of spelling
-- out the monad in the type signature of onIf.

-- XXX maybe make it just [Exp] and add Case to Exp,
-- then we end up accumulating [Case p e]?
-- let ts = onIf t
-- let fs = onIf t
-- mconcat [ts, fs]
-- ...still no way to backpropagate conditons then.

hoistCases' :: Exp -> [(Exp, Exp)]
hoistCases' e = do
  snd $ S.runState (astMap m e) [] -- XXX why does it need Identity here?
  where
    m = ASTMapper { mapOnExp = pure, mapOnIf = onIf }
    -- Want
    --   onExp pred :: Exp -> m Exp
    -- where m is the list monad, so
    --   onExp pred :: Exp -> [Exp]
    -- Hoping to get an expression tree for every predicate.
    onIf :: Exp -> S.StateT [(Exp, Exp)] [] Exp
    onIf (Var x) = pure $ Var x
    onIf (If c t f) = do
      t' <- onIf t
      modify (\s -> (c,t'):s)
      ts <- get
      traceM ("AAY " <> prettyString ts)
      -- let fs = onIf f
      -- traceM $ "t', f' = " <> prettyString ts <> ", " <> prettyString fs
      pure t'
      -- [ts, fs]
    -- onIf p (Cases cases) = do
    --   mconcat $
    --     mapM (\(p', e') -> astMap (m (p :&& p')) e') (NE.toList cases)
    --   -- XXX types check if we mconcat here, figure out what we actually
    --   -- want to do instead. Wait, do we want mconcat?
    --   -- The problem is that we are discarding the predicates!
    onIf v = astMap m v
