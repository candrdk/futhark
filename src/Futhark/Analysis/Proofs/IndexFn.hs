{-# OPTIONS_GHC -Wno-orphans #-}

module Futhark.Analysis.Proofs.IndexFn where

import Data.List.NonEmpty qualified as NE
import Futhark.Analysis.Proofs.Symbol
import Futhark.SoP.SoP (SoP, int2SoP)
import Language.Futhark (VName)

data IndexFn = IndexFn
  { iterator :: Iterator,
    body :: Cases Symbol (SoP Symbol)
  }
  deriving (Show, Eq)

data Domain
  = Iota (SoP Symbol) -- [0, ..., n-1]
  | Cat -- Catenate_{k=0}^{m-1} [b_k, ..., b_{k+1})
      VName -- k
      (SoP Symbol) -- m
      (SoP Symbol) -- b
  deriving (Show, Eq, Ord)

data Iterator
  = Forall VName Domain
  | Empty
  deriving (Show)

instance Eq Iterator where
  (Forall _ u@(Cat k _ _)) == (Forall _ v@(Cat k' _ _)) = u == v && k == k'
  (Forall _ u) == (Forall _ v) = u == v
  Empty == Empty = True
  _ == _ = False

instance Ord Iterator where
  (<=) :: Iterator -> Iterator -> Bool
  _ <= Empty = False
  _ <= Forall {} = True

newtype Cases a b = Cases (NE.NonEmpty (a, b))
  deriving (Show, Eq, Ord)

cases :: [(a, b)] -> Cases a b
cases = Cases . NE.fromList

casesToList :: Cases a b -> [(a, b)]
casesToList (Cases cs) = NE.toList cs

getCase :: Int -> Cases a b -> (a, b)
getCase n (Cases cs) = NE.toList cs !! n

getPredicates :: IndexFn -> [Symbol]
getPredicates (IndexFn _ cs) = map fst $ casesToList cs

getCases :: IndexFn -> Cases Symbol (SoP Symbol)
getCases (IndexFn _ cs) = cs

justSingleCase :: IndexFn -> Maybe (SoP Symbol)
justSingleCase f
  | [(Bool True, f_val)] <- casesToList $ body f =
    Just f_val
  | otherwise =
    Nothing

catVar :: Iterator -> Maybe VName
catVar (Forall _ (Cat k _ _)) = Just k
catVar _ = Nothing

domainSegStart :: Domain -> SoP Symbol
domainSegStart (Iota _) = int2SoP 0
domainSegStart (Cat _ _ b) = b
