module Futhark.Analysis.Proofs.Convert where

import Control.Monad (foldM, forM, unless)
import Control.Monad.RWS
import Data.Bifunctor
import Data.List qualified as L
import Data.List.NonEmpty qualified as NE
import Data.Map qualified as M
import Data.Maybe (fromMaybe, isJust)
import Debug.Trace (traceM)
import Futhark.Analysis.Proofs.IndexFn (Cases (Cases), Domain (..), IndexFn (..), IndexFnM, Iterator (..), VEnv (..), cases, clearAlgEnv, debugPrettyM, insertIndexFn, runIndexFnM, whenDebug, debugM)
import Futhark.Analysis.Proofs.IndexFnPlus (subst)
import Futhark.Analysis.Proofs.Rewrite (rewrite)
import Futhark.Analysis.Proofs.Symbol (Symbol (..), neg)
import Futhark.Analysis.Proofs.Util (prettyBinding)
import Futhark.MonadFreshNames (VNameSource, newVName)
import Futhark.SoP.SoP (SoP, int2SoP, mapSymSoP_, negSoP, sym2SoP, (.*.), (.+.), (.-.), (~+~), (~*~), (~-~))
import Futhark.Util.Pretty (prettyString)
import Language.Futhark qualified as E
import Language.Futhark.Semantic (FileModule (fileProg), ImportName, Imports)
import Futhark.Analysis.Proofs.Unify (unify, Substitution)

--------------------------------------------------------------
-- Extracting information from E.Exp.
--------------------------------------------------------------
getFun :: E.Exp -> Maybe String
getFun (E.Var (E.QualName [] vn) _ _) = Just $ E.baseString vn
getFun _ = Nothing

getSize :: E.Exp -> Maybe (SoP Symbol)
getSize (E.Var _ (E.Info {E.unInfo = ty}) _) = sizeOfTypeBase ty
getSize (E.ArrayLit [] (E.Info {E.unInfo = ty}) _) = sizeOfTypeBase ty
getSize e = error $ "getSize: " <> prettyString e <> "\n" <> show e

sizeOfTypeBase :: E.TypeBase E.Exp as -> Maybe (SoP Symbol)
-- sizeOfTypeBase (E.Scalar (E.Refinement ty _)) =
--   -- TODO why are all refinements scalar?
--   sizeOfTypeBase ty
sizeOfTypeBase (E.Scalar (E.Arrow _ _ _ _ return_type)) =
  sizeOfTypeBase (E.retType return_type)
sizeOfTypeBase (E.Array _ shape _)
  | dim : _ <- E.shapeDims shape =
      Just $ convertSize dim
  where
    convertSize (E.Var (E.QualName _ x) _ _) = sym2SoP $ Var x
    convertSize (E.Parens e _) = convertSize e
    convertSize (E.Attr _ e _) = convertSize e
    convertSize (E.IntLit x _ _) = int2SoP x
    convertSize e = error ("convertSize not implemented for: " <> show e)
sizeOfTypeBase _ = Nothing

-- Strip unused information.
getArgs :: NE.NonEmpty (a, E.Exp) -> [E.Exp]
getArgs = map (stripExp . snd) . NE.toList
  where
    stripExp x = fromMaybe x (E.stripExp x)

--------------------------------------------------------------
-- Construct index function for source program
--------------------------------------------------------------
mkIndexFnProg :: VNameSource -> Imports -> M.Map E.VName IndexFn
mkIndexFnProg vns prog = snd $ runIndexFnM (mkIndexFnImports prog) vns

mkIndexFnImports :: [(ImportName, FileModule)] -> IndexFnM ()
mkIndexFnImports = mapM_ (mkIndexFnDecs . E.progDecs . fileProg . snd)

-- A program is a list of declarations (DecBase); functions are value bindings
-- (ValBind). Everything is in an AppExp.

mkIndexFnDecs :: [E.Dec] -> IndexFnM ()
mkIndexFnDecs [] = pure ()
mkIndexFnDecs (E.ValDec vb : rest) = do
  _ <- mkIndexFnValBind vb
  mkIndexFnDecs rest
mkIndexFnDecs (_ : ds) = mkIndexFnDecs ds

-- toplevel_indexfns
mkIndexFnValBind :: E.ValBind -> IndexFnM (Maybe IndexFn)
mkIndexFnValBind val@(E.ValBind _ vn _ret _ _ _params body _ _ _) = do
  clearAlgEnv
  debugPrettyM "\n====\nmkIndexFnValBind:\n\n" val
  indexfn <- forward body >>= refineAndBind vn
  -- insertTopLevel vn (params, indexfn)
  algenv <- gets algenv
  debugPrettyM "mkIndexFnValBind AlgEnv\n" algenv
  pure (Just indexfn)

refineAndBind :: E.VName -> IndexFn -> IndexFnM IndexFn
refineAndBind vn indexfn = do
  indexfn' <- rewrite indexfn
  insertIndexFn vn indexfn'
  whenDebug (traceM $ prettyBinding vn indexfn')
  -- tell ["resulting in", toLaTeX (vn, indexfn')]
  pure indexfn'

singleCase :: a -> Cases Symbol a
singleCase e = cases [(Bool True, e)]

fromScalar :: SoP Symbol -> IndexFn
fromScalar e = IndexFn Empty (singleCase e)

forward :: E.Exp -> IndexFnM IndexFn
forward (E.Parens e _) = forward e
forward (E.Attr _ e _) = forward e
-- Let-bindings.
forward (E.AppExp (E.LetPat _ (E.Id vn _ _) x body _) _) = do
  -- tell [textbf "Forward on " <> Math.math (toLaTeX vn) <> toLaTeX x]
  _ <- refineAndBind vn =<< forward x
  forward body
-- Tuples left unhandled for now.
-- forward (E.AppExp (E.LetPat _ p@(E.TuplePat patterns _) x body _) _) = do
--     -- tell [textbf "Forward on " <> Math.math (toLaTeX (S.toList $ E.patNames p)) <> toLaTeX x]
--     xs <- unzipT <$> forward x
--     forM_ (zip patterns xs) refineAndBind'
--     forward body
--     where
--       -- Wrap refineAndBind to discard results otherwise bound to wildcards.
--       refineAndBind' (E.Wildcard {}, _) = pure ()
--       refineAndBind' (E.Id vn _ _, indexfn) =
--         void (refineAndBind vn indexfn)
--       refineAndBind' e = error ("not implemented for " <> show e)
-- Leaves.
forward (E.Literal (E.BoolValue x) _) =
  pure . fromScalar . sym2SoP $ Bool x
forward (E.Literal (E.SignedValue (E.Int64Value x)) _) =
  pure . fromScalar . int2SoP $ toInteger x
forward (E.IntLit x _ _) =
  pure . fromScalar $ int2SoP x
forward (E.Negate (E.IntLit x _ _) _) =
  pure . fromScalar . negSoP $ int2SoP x
forward e@(E.Var (E.QualName _ vn) _ _) = do
  indexfns <- gets indexfns
  case M.lookup vn indexfns of
    Just indexfn -> do
      pure indexfn
    _ -> do
      -- TODO handle refinement types
      -- handleRefinementTypes e
      case getSize e of
        Just sz -> do
          -- Canonical array representation.
          i <- newVName "i"
          rewrite $
            IndexFn
              (Forall i (Iota sz))
              (singleCase . sym2SoP $ Idx (Var vn) (sym2SoP $ Var i))
        Nothing ->
          -- Canonical scalar representation.
          rewrite $ IndexFn Empty (singleCase . sym2SoP $ Var vn)
-- Nodes.
-- TODO handle tuples later.
-- forward (E.TupLit es _) = do
--   xs <- mapM forward es
--   vns <- mapM (\_ -> newVName "xs") xs
--   let IndexFn iter1 _ = head xs
--   foldM (\acc (vn, x) -> sub vn x acc)
--         (IndexFn iter1 (toCases . Tuple $ map Var vns))
--         (zip vns xs)
--     >>= rewrite
forward (E.AppExp (E.Index xs' slice _) _)
  | [E.DimFix idx'] <- slice = do
      -- XXX support only simple indexing for now
      IndexFn iter_idx idx <- forward idx'
      IndexFn iter_xs xs <- forward xs'
      case iter_xs of
        Forall j _ -> do
          subst j (IndexFn iter_idx idx) (IndexFn iter_idx xs)
        _ ->
          error "indexing into a scalar"
forward (E.Not e _) = do
  IndexFn it e' <- forward e
  rewrite $ IndexFn it $ cmapValues (mapSymSoP_ neg) e'
forward (E.AppExp (E.BinOp (op', _) _ (x', _) (y', _) _) _)
  | E.baseTag (E.qualLeaf op') <= E.maxIntrinsicTag,
    name <- E.baseString $ E.qualLeaf op',
    Just bop <- L.find ((name ==) . prettyString) [minBound .. maxBound :: E.BinOp] = do
      vx <- forward x'
      let IndexFn iter_x _ = vx
      vy <- forward y'
      a <- newVName "a"
      b <- newVName "b"
      let doOp op =
            subst a vx (IndexFn iter_x (singleCase $ op (Var a) (Var b)))
              >>= subst b vy
              >>= rewrite
      case bop of
        E.Plus -> doOp (~+~)
        E.Times -> doOp (~*~)
        E.Minus -> doOp (~-~)
        E.Equal -> doOp (~==~)
        E.Less -> doOp (~<~)
        E.Greater -> doOp (~>~)
        E.Leq -> doOp (~<=~)
        E.LogAnd -> doOp (~&&~)
        E.LogOr -> doOp (~||~)
        _ -> error ("forward not implemented for bin op: " <> show bop)
forward (E.AppExp (E.If c t f _) _) = do
  IndexFn iter_c c' <- forward c
  vt <- forward t
  vf <- forward f
  -- Negating `c` means negating the case _values_ of c, keeping the
  -- conditions of any nested if-statements (case conditions) untouched.
  cond <- newVName "cond"
  t_branch <- newVName "t_branch"
  f_branch <- newVName "f_branch"
  let y =
        IndexFn
          iter_c
          ( cases
              [ (Var cond, sym2SoP $ Var t_branch),
                (neg $ Var cond, sym2SoP $ Var f_branch)
              ]
          )
  subst cond (IndexFn iter_c c') y
    >>= subst t_branch vt
    >>= subst f_branch vf
    >>= rewrite
-- forward e | trace ("forward\n  " ++ prettyString e) False =
--   -- All calls after this case get traced.
--   undefined
forward expr@(E.AppExp (E.Apply f args _) _)
  | Just fname <- getFun f,
    "map" `L.isPrefixOf` fname,
    E.Lambda params body _ _ _ : args' <- getArgs args = do
      xss <- mapM forward args'
      debugPrettyM "map args:" xss
      let IndexFn iter_first_arg _ = head xss
      -- TODO use iter_body; likely needed for nested maps?
      IndexFn iter_body cases_body <- forward body
      unless
        (iter_body == iter_first_arg || iter_body == Empty)
        ( error $
            "map internal error: iter_body != iter_first_arg"
              <> show iter_body
              <> show iter_first_arg
        )
      -- Make susbtitutions from function arguments to array names.
      let paramNames :: [E.VName] = concatMap E.patNames params
      -- TODO handle tupled values by splitting them into separate index functions
      -- let xss_flat :: [IndexFn] = mconcat $ map unzipT xss
      let xss_flat = xss
      let y' = IndexFn iter_first_arg cases_body
      -- tell ["Using map rule ", toLaTeX y']
      foldM substParams y' (zip paramNames xss_flat)
        >>= rewrite
  | Just fname <- getFun f,
    "map" `L.isPrefixOf` fname = do
      -- No need to handle map non-lambda yet as program can just be rewritten.
      error $
        "forward on map with non-lambda function arg: "
          <> prettyString expr
          <> ". Eta-expand your program."
  | Just "replicate" <- getFun f,
    [n, x] <- getArgs args = do
      debugM "replicate n x"
      n' <- forward n
      debugPrettyM "n" n'
      x' <- forward x
      debugPrettyM "x" x'
      i <- newVName "i"
      case (n', x') of
        ( IndexFn Empty (Cases ((Bool True, m) NE.:| [])),
          IndexFn Empty body
          ) ->
            -- XXX support only 1D arrays for now.
            rewrite $ IndexFn (Forall i (Iota m)) body
        _ -> undefined -- TODO See iota comment.
  | Just "iota" <- getFun f,
    [n] <- getArgs args = do
      indexfn <- forward n
      i <- newVName "i"
      case indexfn of
        IndexFn Empty (Cases ((Bool True, m) NE.:| [])) ->
          rewrite $ IndexFn (Forall i (Iota m)) (singleCase . sym2SoP $ Var i)
        _ -> undefined -- TODO We've no way to express this yet.
        -- Have talked with Cosmin about an "outer if" before.
  | Just "scan" <- getFun f,
    [E.OpSection (E.QualName [] vn) _ _, _ne, xs'] <- getArgs args = do
      -- Scan with basic operator.
      IndexFn iter_xs xs <- forward xs'
      let i = case iter_xs of
            (Forall i' _) -> i'
            Empty -> error "scan array is empty?"
      -- TODO should verify that _ne matches op
      op <-
        case E.baseString vn of
          "+" -> pure (~+~)
          "-" -> pure (~-~)
          "*" -> pure (~*~)
          _ -> error ("scan not implemented for bin op: " <> show vn)
      let base_case = sym2SoP (Var i) :== int2SoP 0
      x <- newVName "a"
      let y =
            IndexFn
              iter_xs
              ( cases
                  [(base_case, sym2SoP (Var x)), (neg base_case, Recurrence `op` Var x)]
              )
      -- tell ["Using scan rule ", toLaTeX y]
      subst x (IndexFn iter_xs xs) y
        >>= rewrite
  | Just "scatter" <- getFun f,
    [dest_arg, inds_arg, vals_arg] <- getArgs args = do
    -- Scatter in-bounds-monotonic indices.
    --
    -- y = scatter dest inds vals
    -- where
    --   inds = ∀k ∈ [0, ..., m-1] .
    --       | c(k)  => e1(k)
    --       | ¬c(k) => OOB
    --   xs is an array of size at least m
    --   e1(k-1) <= e1(k) for all k
    --   dest has size e1(m-1)         (to ensure conclusion covers all of dest)
    --   e1(0) is 0
    --   OOB < 0 or OOB >= e1(m-1)
    -- ___________________________________________________
    -- y = ∀i ∈ ⊎k=iota m [e1(k), ..., e1(k+1)] .
    --     | i == inds[k] => vals[k]
    --     | i /= inds[k] => dest[i]
    --
    -- Note that case predicates c and ¬c from inds are not propagated by y.
    -- Leaving them out is safe, because i == inds[k] only if inds[k] == xs[k],
    -- which in turn implies c. Similarly for ¬c.
    --
    -- From type checking, we have:
    -- scatter : (dest : [n]t) -> (inds : [m]i64) -> (vals : [m]t) : [n]t
    -- * inds and vals are same size
    -- * dest and result are same size
    dest <- forward dest_arg
    inds <- forward inds_arg
    vals <- forward vals_arg
    -- 1. Check that inds is on the right form.
    --    1.i. Extract m from inds.
    --    1.ii. Extract (xs, OOB) from inds.
    --          This requires determining which is which.
    --    All of this should be doable with unification?
    tmp_k <- newVName "k"
    tmp_m <- newVName "m"
    tmp_e1 <- newVName "e1"
    tmp_OOB <- newVName "OOB"
    tmp_c <- newVName "OOB"
    tmp_negc <- newVName "OOB"
    let inds_template =
          IndexFn {
            iterator = Forall tmp_k (Iota $ sym2SoP $ Hole tmp_m),
            body = cases [(Hole tmp_c, sym2SoP $ Hole tmp_e1),
                          (Hole tmp_negc, sym2SoP $ Hole tmp_OOB)]
            }
    debugPrettyM "inds" inds
    s :: Maybe (Substitution Symbol) <- unify inds_template inds
    unless (isJust s) (error "unhandled scatter")
    -- 3. Check that xs has size at least m. (Why?)
    -- 4. Check that xs[0] = 0.
    -- 5. Check that xs is monotonically increasing for k in [0, ..., m-1].
    --    Use query to solver, so decide on interface. Decide on translation
    --    to solver-Symbol. Decide on how "monotonically increasing" translates
    --    to a query.
    -- 6. Check that OOB < 0 or OOB >= xs[m-1].
    error "scatter not implemented yet"
  -- Applying other functions, for instance, user-defined ones.
  | (E.Var (E.QualName [] g) info _) <- f,
    args' <- getArgs args,
    E.Scalar (E.Arrow _ _ _ _ (E.RetType _ return_type)) <- E.unInfo info = do
      toplevel <- gets toplevel
      case M.lookup g toplevel of
        Just (_param_names, _param_sizes, _indexfn) ->
          -- g is a previously analyzed user-defined top-level function.
          error "use of top-level defs not implemented yet"
        Nothing -> do
          -- g is a free variable in this expression (probably a parameter
          -- to the top-level function currently being analyzed).
          params <- mapM forward args'
          param_names <- forM params (const $ newVName "x")
          iter <-
            case sizeOfTypeBase return_type of
              Just sz -> do
                -- Function returns an array.
                i <- newVName "i"
                pure $ Forall i (Iota sz)
              Nothing -> do
                pure Empty
          let g_fn =
                IndexFn
                  { iterator = iter,
                    body =
                      singleCase . sym2SoP $
                        Apply (Var g) (map (sym2SoP . Var) param_names)
                  }
          debugPrettyM "g_fn:" g_fn
          foldM substParams g_fn (zip param_names params)
            >>= rewrite
forward e = error $ "forward on " <> show e

substParams :: IndexFn -> (E.VName, IndexFn) -> IndexFnM IndexFn
substParams y (paramName, paramIndexFn) =
  subst paramName paramIndexFn y >>= rewrite

cmap :: ((a, b) -> (c, d)) -> Cases a b -> Cases c d
cmap f (Cases xs) = Cases (fmap f xs)

cmapValues :: (b -> c) -> Cases a b -> Cases a c
cmapValues f = cmap (second f)

-- TODO eh bad
(~==~) :: Symbol -> Symbol -> SoP Symbol
x ~==~ y = sym2SoP $ sym2SoP x :== sym2SoP y

(~<~) :: Symbol -> Symbol -> SoP Symbol
x ~<~ y = sym2SoP $ sym2SoP x :< sym2SoP y

(~>~) :: Symbol -> Symbol -> SoP Symbol
x ~>~ y = sym2SoP $ sym2SoP x :> sym2SoP y

(~<=~) :: Symbol -> Symbol -> SoP Symbol
x ~<=~ y = sym2SoP $ sym2SoP x :<= sym2SoP y

(~&&~) :: Symbol -> Symbol -> SoP Symbol
x ~&&~ y = sym2SoP $ x :&& y

(~||~) :: Symbol -> Symbol -> SoP Symbol
x ~||~ y = sym2SoP $ x :|| y
