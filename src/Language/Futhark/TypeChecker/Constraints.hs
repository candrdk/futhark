module Language.Futhark.TypeChecker.Constraints
  ( Type,
    toType,
    Ct (..),
    Constraints,
    TyVarInfo (..),
    TyVar,
    TyVars,
    solve,
  )
where

import Control.Monad.Except
import Control.Monad.State
import Data.Bifunctor
import Data.Map qualified as M
import Data.Text qualified as T
import Futhark.Util.Pretty
import Language.Futhark

-- | A shape component is currently just unit. The rank of an array is
-- then just the number of shape components it contains in its shape
-- list. When we add AUTOMAP, these components will also allow shape
-- variables. The list of components should then be understood as
-- concatenation of shapes (meaning you can't just take the length to
-- determine the rank of the array).
type SComp = ()

-- | The type representation used by the constraint solver. Agnostic
-- to sizes.
type Type = TypeBase SComp NoUniqueness

toType :: TypeBase d u -> Type
toType = bimap (const ()) (const NoUniqueness)

data Ct = CtEq Type Type
  deriving (Show)

instance Pretty Ct where
  pretty (CtEq t1 t2) = pretty t1 <+> "~" <+> pretty t2

type Constraints = [Ct]

-- | Information about a type variable.
data TyVarInfo
  = -- | Can be substituted with anything.
    TyVarFree
  | -- | Can only be substituted with these primitive types.
    TyVarPrim [PrimType]
  | -- | Must be a record with these fields.
    TyVarRecord (M.Map Name Type)
  | -- | Must be a sum type with these fields.
    TyVarSum (M.Map Name [Type])
  deriving (Show)

instance Pretty TyVarInfo where
  pretty TyVarFree = "free"
  pretty (TyVarPrim pts) = "∈" <+> pretty pts
  pretty (TyVarRecord fs) = pretty $ Scalar $ Record fs
  pretty (TyVarSum cs) = pretty $ Scalar $ Sum cs

type TyVar = VName

-- | If a VName is not in this map, it is assumed to be rigid.
type TyVars = M.Map TyVar TyVarInfo

data TyVarSol
  = -- | Has been substituted with this.
    TyVarSol Type
  | -- | Replaced by this other type variable.
    TyVarLink VName
  | -- | Not substituted yet; has this constraint.
    TyVarUnsol TyVarInfo
  deriving (Show)

newtype SolverState = SolverState {solverTyVars :: M.Map TyVar TyVarSol}

initialState :: TyVars -> SolverState
initialState tyvars = SolverState $ M.map TyVarUnsol tyvars

substTyVars :: (Monoid u) => M.Map TyVar TyVarSol -> TypeBase SComp u -> TypeBase SComp u
substTyVars m t@(Scalar (TypeVar u (QualName qs v) args)) =
  case M.lookup v m of
    Just (TyVarLink v') ->
      substTyVars m $ Scalar $ TypeVar u (QualName qs v') args
    Just (TyVarSol t') -> second (const mempty) t'
    Just (TyVarUnsol _) -> t
    Nothing -> t
substTyVars _ (Scalar (Prim pt)) = Scalar $ Prim pt
substTyVars m (Scalar (Record fs)) = Scalar $ Record $ M.map (substTyVars m) fs
substTyVars m (Scalar (Sum cs)) = Scalar $ Sum $ M.map (map $ substTyVars m) cs
substTyVars m (Scalar (Arrow u pname d t1 (RetType ext t2))) =
  Scalar $ Arrow u pname d (substTyVars m t1) $ RetType ext $ substTyVars m t2
substTyVars m (Array u shape elemt) =
  arrayOfWithAliases u shape $ substTyVars m $ Scalar elemt

solution :: SolverState -> M.Map TyVar Type
solution s = M.mapMaybe f $ solverTyVars s
  where
    f (TyVarSol t) = Just $ substTyVars (solverTyVars s) t
    f (TyVarLink v) = f =<< M.lookup v (solverTyVars s)
    f (TyVarUnsol _) = Nothing

newtype SolveM a = SolveM {runSolveM :: StateT SolverState (Except T.Text) a}
  deriving (Functor, Applicative, Monad, MonadState SolverState, MonadError T.Text)

subTyVar :: VName -> Type -> SolveM ()
subTyVar v t =
  modify $ \s -> s {solverTyVars = M.insert v (TyVarSol t) $ solverTyVars s}

linkTyVar :: VName -> VName -> SolveM ()
linkTyVar v t =
  modify $ \s -> s {solverTyVars = M.insert v (TyVarLink t) $ solverTyVars s}

-- Unify at the root, emitting new equalities that must hold.
unify :: Type -> Type -> Maybe [(Type, Type)]
unify (Scalar (Prim pt1)) (Scalar (Prim pt2))
  | pt1 == pt2 = Just []
unify (Scalar (Arrow _ _ _ t1a (RetType _ t1r))) (Scalar (Arrow _ _ _ t2a (RetType _ t2r))) =
  Just [(t1a, t2a), (toType t1r, toType t2r)]
unify (Scalar (Record fs1)) (Scalar (Record fs2))
  | M.keys fs1 == M.keys fs2 =
      Just $ M.elems $ M.intersectionWith (,) fs1 fs2
unify (Scalar (Sum cs1)) (Scalar (Sum cs2))
  | M.keys cs1 == M.keys cs2 = do
      fmap concat
        . forM (M.elems $ M.intersectionWith (,) cs1 cs2)
        $ \(ts1, ts2) -> do
          guard $ length ts1 == length ts2
          Just $ zip ts1 ts2
unify t1 t2
  | Just t1' <- peelArray 1 t1,
    Just t2' <- peelArray 1 t2 =
      Just [(t1', t2')]
unify _ _ = Nothing

solveCt :: Ct -> SolveM ()
solveCt ct = do
  let CtEq t1 t2 = ct
  solveCt' (t1, t2)
  where
    bad = throwError $ "Unsolvable: " <> prettyText ct
    solveCt' (t1, t2) = do
      tyvars <- gets solverTyVars
      let flexible v = case M.lookup v tyvars of
            Just (TyVarLink v') -> flexible v'
            Just (TyVarUnsol _) -> True
            Just (TyVarSol _) -> False
            Nothing -> False
          sub t@(Scalar (TypeVar u (QualName [] v) [])) =
            case M.lookup v tyvars of
              Just (TyVarLink v') -> sub $ Scalar (TypeVar u (QualName [] v') [])
              Just (TyVarSol t') -> sub t'
              _ -> t
          sub t = t
      case (sub t1, sub t2) of
        ( t1'@(Scalar (TypeVar _ (QualName [] v1) [])),
          t2'@(Scalar (TypeVar _ (QualName [] v2) []))
          )
            | v1 == v2 -> pure ()
            | otherwise ->
                case (flexible v1, flexible v2) of
                  (False, False) -> bad
                  (True, False) -> subTyVar v1 t2'
                  (False, True) -> subTyVar v2 t1'
                  (True, True) -> linkTyVar v1 v2
        (Scalar (TypeVar _ (QualName [] v1) []), t2') ->
          if flexible v1 then subTyVar v1 t2' else bad
        (t1', Scalar (TypeVar _ (QualName [] v2) [])) ->
          if flexible v2 then subTyVar v2 t1' else bad
        (t1', t2') -> case unify t1' t2' of
          Nothing -> bad
          Just eqs -> mapM_ solveCt' eqs

solve :: Constraints -> TyVars -> Either T.Text (M.Map TyVar Type)
solve constraints tyvars =
  second solution
    . runExcept
    . flip execStateT (initialState tyvars)
    . runSolveM
    $ mapM solveCt constraints
{-# NOINLINE solve #-}
