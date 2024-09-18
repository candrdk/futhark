{-# OPTIONS_GHC -Wno-orphans #-}

module Futhark.Analysis.Proofs.IndexFn where

import Control.Monad.RWS.Strict
import Data.List.NonEmpty qualified as NE
import Data.Map qualified as M
import Futhark.Analysis.Proofs.Symbol
import Futhark.MonadFreshNames
import Futhark.SoP.Monad (AlgEnv (..), MonadSoP (..), Nameable (mkName))
import Futhark.SoP.SoP (SoP)
import Language.Futhark (VName)
import Language.Futhark qualified as E

data IndexFn = IndexFn
  { iterator :: Iterator,
    body :: Cases Symbol (SoP Symbol)
  }
  deriving (Show)

data Domain
  = Iota (SoP Symbol) -- [0, ..., n-1]
  | Cat -- Catenate_{k=1}^{m-1} [b_{k-1}, ..., b_k)
      VName -- k
      (SoP Symbol) -- m
      (SoP Symbol) -- b
  deriving (Show)

data Iterator
  = Forall VName Domain
  | Empty
  deriving (Show)

data Cases a b
  = Cases (NE.NonEmpty (a, b))
  | CHole VName
  deriving (Show, Eq, Ord)

cases :: [(a, b)] -> Cases a b
cases = Cases . NE.fromList

casesToList :: Cases a b -> [(a, b)]
casesToList (Cases cs) = NE.toList cs
casesToList (CHole _) = error "casesToList on hole."

-------------------------------------------------------------------------------
-- Monad.
-------------------------------------------------------------------------------
data VEnv = VEnv
  { vnamesource :: VNameSource,
    algenv :: AlgEnv Symbol E.Exp Property,
    indexfns :: M.Map VName IndexFn
    -- toplevel :: M.Map E.VName ([E.Pat], IndexFn)
  }

newtype IndexFnM a = IndexFnM (RWS () () VEnv a)
  deriving
    ( Applicative,
      Functor,
      Monad,
      MonadFreshNames,
      MonadState VEnv
    )

instance (Monoid w) => MonadFreshNames (RWS r w VEnv) where
  getNameSource = gets vnamesource
  putNameSource vns = modify $ \senv -> senv {vnamesource = vns}

-- This is required by MonadSoP.
instance Nameable Symbol where
  mkName (VNameSource i) = (Var $ E.VName "x" i, VNameSource $ i + 1)

instance MonadSoP Symbol E.Exp Property IndexFnM where
  getUntrans = gets (untrans . algenv)
  getRanges = gets (ranges . algenv)
  getEquivs = gets (equivs . algenv)
  modifyEnv f = modify $ \env -> env {algenv = f $ algenv env}

runIndexFnM :: IndexFnM a -> VNameSource -> (a, M.Map VName IndexFn)
runIndexFnM (IndexFnM m) vns = getRes $ runRWS m () s
  where
    getRes (x, env, _) = (x, indexfns env)
    s = VEnv vns mempty mempty

insertIndexFn :: E.VName -> IndexFn -> IndexFnM ()
insertIndexFn x v =
  modify $ \env -> env {indexfns = M.insert x v $ indexfns env}

-- insertTopLevel :: E.VName -> ([E.Pat], IndexFn) -> IndexFnM ()
-- insertTopLevel vn (args, ixfn) =
--   modify $
--     \env -> env {toplevel = M.insert vn (args, ixfn) $ toplevel env}

clearAlgEnv :: IndexFnM ()
clearAlgEnv =
  modify $ \env -> env {algenv = mempty}
