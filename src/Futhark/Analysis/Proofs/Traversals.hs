module Futhark.Analysis.Proofs.Traversals
where

import Futhark.Analysis.Proofs.Symbol
import Futhark.SoP.SoP (SoP, (.+.), int2SoP, sopToLists, (.*.), sym2SoP)
import Futhark.Analysis.Proofs.IndexFn (IndexFn (..), Domain (..), Iterator (..), Cases (..), casesToList, cases)
import Debug.Trace (trace)

data ASTMapper a m = ASTMapper
  { mapOnSymbol :: a -> m a,
    mapOnSoP :: SoP a -> m (SoP a)
  }

class ASTMappable a b where
  astMap :: (Monad m) => ASTMapper a m -> b -> m b

instance Ord a => ASTMappable a (SoP a) where
  astMap m sop = do
    mapOnSoP m . foldl (.+.) (int2SoP 0) =<< mapM g (sopToLists sop)
    where
      g (ts, c) = do
        ts' <- mapM (mapOnSymbol m) ts
        pure $ foldl (.*.) (int2SoP 1) (int2SoP c : map sym2SoP ts')

instance ASTMappable Symbol Symbol where
  astMap _ Recurrence = pure Recurrence
  astMap m (Var x) = mapOnSymbol m $ Var x
  astMap m (Hole x) = mapOnSymbol m $ Hole x
  astMap m (LinComb vn lb ub x) =
    mapOnSymbol m =<< LinComb vn <$> astMap m lb <*> astMap m ub <*> astMap m x
  astMap m (Idx xs i) = mapOnSymbol m =<< Idx <$> astMap m xs <*> astMap m i
  astMap m (Indicator p) = mapOnSymbol m . Indicator =<< astMap m p
  astMap _ x@(Bool {}) = pure x
  astMap m (Not x) = mapOnSymbol m . Not =<< astMap m x
  astMap m (x :== y) = mapOnSymbol m =<< (:==) <$> astMap m x <*> astMap m y
  astMap m (x :< y) = mapOnSymbol m =<< (:<) <$> astMap m x <*> astMap m y
  astMap m (x :> y) = mapOnSymbol m =<< (:>) <$> astMap m x <*> astMap m y
  astMap m (x :/= y) = mapOnSymbol m =<< (:/=) <$> astMap m x <*> astMap m y
  astMap m (x :>= y) = mapOnSymbol m =<< (:>=) <$> astMap m x <*> astMap m y
  astMap m (x :<= y) = mapOnSymbol m =<< (:<=) <$> astMap m x <*> astMap m y
  astMap m (x :&& y) = mapOnSymbol m =<< (:&&) <$> astMap m x <*> astMap m y
  astMap m (x :|| y) = mapOnSymbol m =<< (:||) <$> astMap m x <*> astMap m y

instance ASTMappable Symbol IndexFn where
  astMap m (IndexFn dom body) = IndexFn <$> astMap m dom <*> astMap m body

instance ASTMappable Symbol Iterator where
  astMap m (Forall i dom) = Forall i <$> astMap m dom
  astMap _ Empty = pure Empty

instance ASTMappable Symbol Domain where
  astMap m (Iota n) = Iota <$> astMap m n
  astMap m (Cat k n b) = Cat k <$> astMap m n <*> astMap m b

instance ASTMappable Symbol (Cases Symbol (SoP Symbol)) where
  astMap _ cs | trace ("astMap " <> show cs) False = undefined
  astMap m cs = do
    let (ps, qs) = unzip $ casesToList cs
    ps' <- mapM (astMap m) ps
    qs' <- mapM (astMap m) qs
    pure . cases $ zip ps' qs'
