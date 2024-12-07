{-# LANGUAGE TypeFamilies #-}

module Futhark.AD.Fwd (fwdJVP) where

import Control.Monad
import Control.Monad.Reader
import Control.Monad.State.Strict
import Data.Bifunctor (second)
import Data.List (transpose)
import Data.List.NonEmpty (NonEmpty (..))
import Data.Map qualified as M
import Futhark.AD.Derivatives
import Futhark.Analysis.PrimExp.Convert
import Futhark.Builder
import Futhark.Construct
import Futhark.IR.SOACS

zeroTan :: Type -> ADM SubExp
zeroTan (Prim t) = pure $ constant $ blankPrimValue t
zeroTan t = error $ "zeroTan on non-primitive type: " ++ prettyString t

zeroExp :: Type -> Exp SOACS
zeroExp (Prim pt) =
  BasicOp $ SubExp $ Constant $ blankPrimValue pt
zeroExp (Array pt shape _) =
  BasicOp $ Replicate shape $ Constant $ blankPrimValue pt
zeroExp t = error $ "zeroExp: " ++ show t

tanType :: (ArrayShape s, Monoid u) => TypeBase s u -> ADM (TypeBase s u)
tanType (Acc acc ispace ts u) = do
  ts_tan <- mapM tanType ts
  pure $ Acc acc ispace (ts ++ ts_tan) u
tanType t = do
  shape <- askShape
  pure $
    arrayOf
      (Prim (elemType t))
      (shape `prependShape` arrayShape t)
      (uniqueness t)

slocal' :: ADM a -> ADM a
slocal' = slocal id

slocal :: (RState -> RState) -> ADM a -> ADM a
slocal f m = do
  s <- get
  modify f
  a <- m
  modify $ \s' -> s' {stateTans = stateTans s}
  pure a

data RState = RState
  { stateTans :: M.Map VName VName,
    stateNameSource :: VNameSource
  }

newtype ADM a = ADM (BuilderT SOACS (ReaderT Shape (State RState)) a)
  deriving
    ( Functor,
      Applicative,
      Monad,
      MonadState RState,
      MonadFreshNames,
      HasScope SOACS,
      LocalScope SOACS
    )

instance MonadBuilder ADM where
  type Rep ADM = SOACS
  mkExpDecM pat e = ADM $ mkExpDecM pat e
  mkBodyM bnds res = ADM $ mkBodyM bnds res
  mkLetNamesM pat e = ADM $ mkLetNamesM pat e

  addStms = ADM . addStms
  collectStms (ADM m) = ADM $ collectStms m

instance MonadFreshNames (State RState) where
  getNameSource = gets stateNameSource
  putNameSource src = modify (\env -> env {stateNameSource = src})

askShape :: ADM Shape
askShape = ADM $ lift ask

runADM :: (MonadFreshNames m) => Shape -> ADM a -> m a
runADM shape (ADM m) =
  modifyNameSource $ \vn ->
    second stateNameSource $
      runState
        (runReaderT (fst <$> runBuilderT m mempty) shape)
        (RState mempty vn)

tanVName :: VName -> ADM VName
tanVName v = newVName (baseString v <> "_tan")

insertTan :: VName -> VName -> ADM ()
insertTan v v' =
  modify $ \env -> env {stateTans = M.insert v v' (stateTans env)}

class TanBuilder a where
  newTan :: a -> ADM a
  bundleNew :: a -> ADM [a]

bundleNewList :: (TanBuilder a) => [a] -> ADM [a]
bundleNewList = fmap mconcat . mapM bundleNew

instance (ArrayShape s, Monoid u) => TanBuilder (PatElem (TypeBase s u)) where
  newTan (PatElem p t)
    | isAcc t = do
        insertTan p p
        t' <- tanType t
        pure $ PatElem p t'
    | otherwise = do
        p' <- tanVName p
        insertTan p p'
        t' <- tanType t
        pure $ PatElem p' t'
  bundleNew pe@(PatElem _ t) = do
    pe' <- newTan pe
    if isAcc t
      then pure [pe']
      else pure [pe, pe']

newTanPat :: (TanBuilder (PatElem t)) => Pat t -> ADM (Pat t)
newTanPat (Pat pes) = Pat <$> mapM newTan pes

bundleNewPat :: (TanBuilder (PatElem t)) => Pat t -> ADM (Pat t)
bundleNewPat (Pat pes) = Pat <$> bundleNewList pes

instance (ArrayShape s, Monoid u) => TanBuilder (Param (TypeBase s u)) where
  newTan (Param _ p t) = do
    PatElem p' t' <- newTan $ PatElem p t
    pure $ Param mempty p' t'
  bundleNew param@(Param _ _ (Prim Unit)) =
    pure [param]
  bundleNew param@(Param _ _ t) = do
    param' <- newTan param
    if isAcc t
      then pure [param']
      else pure [param, param']

instance
  (ArrayShape s, Monoid u, Tangent a) =>
  TanBuilder (Param (TypeBase s u), a)
  where
  newTan (p, x) = (,) <$> newTan p <*> tangent x
  bundleNew (p, x) = do
    b <- bundleNew p
    x_tan <- tangent x
    pure $ zip b [x, x_tan]

class Tangent a where
  tangent :: a -> ADM a
  bundleTan :: a -> ADM [a]

instance (ArrayShape s, Monoid u) => Tangent (TypeBase s u) where
  tangent = tanType
  bundleTan t
    | isAcc t = do
        t' <- tangent t
        pure [t']
    | otherwise = do
        t' <- tangent t
        pure [t, t']

bundleTangents :: (Tangent a) => [a] -> ADM [a]
bundleTangents = (mconcat <$>) . mapM bundleTan

instance Tangent VName where
  tangent v = do
    maybeTan <- gets $ M.lookup v . stateTans
    case maybeTan of
      Just v_tan -> pure v_tan
      Nothing -> do
        t <- lookupType v
        letExp (baseString v <> "_implicit_tan") $ zeroExp t
  bundleTan v = do
    t <- lookupType v
    if isAcc t
      then pure [v]
      else do
        v_tan <- tangent v
        pure [v, v_tan]

instance Tangent SubExp where
  tangent (Constant c) = zeroTan $ Prim $ primValueType c
  tangent (Var v) = Var <$> tangent v
  bundleTan c@Constant {} = do
    c_tan <- tangent c
    pure [c, c_tan]
  bundleTan (Var v) = fmap Var <$> bundleTan v

instance Tangent SubExpRes where
  tangent (SubExpRes cs se) = SubExpRes cs <$> tangent se
  bundleTan (SubExpRes cs se) = map (SubExpRes cs) <$> bundleTan se

asVName :: SubExp -> ADM VName
asVName (Var v) = pure v
asVName (Constant x) = letExp "v" $ BasicOp $ SubExp $ Constant x

withTan ::
  SubExp ->
  (SubExp -> ADM (Exp SOACS)) ->
  ADM (Exp SOACS)
withTan x f = do
  shape <- askShape
  x_tan <- tangent x
  if shape == mempty
    then f x_tan
    else do
      let w = shapeSize 0 shape
      x_tan_v <- asVName x_tan
      x_tan_p <- newParam "x_tanp" . rowType =<< lookupType x_tan_v
      lam <- mkLambda [x_tan_p] $ do
        fmap (subExpsRes . pure) . letSubExp "tan"
          =<< f (Var (paramName x_tan_p))
      pure $ Op $ Screma w [x_tan_v] (mapSOAC lam)

withTans ::
  PrimType ->
  SubExp ->
  SubExp ->
  (PrimExp VName -> PrimExp VName -> PrimExp VName) ->
  ADM (Exp SOACS)
withTans t x y f = do
  shape <- askShape
  x_tan <- asVName =<< tangent x
  y_tan <- asVName =<< tangent y
  if shape == mempty
    then toExp $ f (LeafExp x_tan t) (LeafExp y_tan t)
    else do
      let w = shapeSize 0 shape
      x_tan_p <- newParam "x_tanp" . rowType =<< lookupType x_tan
      y_tan_p <- newParam "y_tanp" . rowType =<< lookupType y_tan
      lam <- mkLambda [x_tan_p, y_tan_p] $ do
        fmap (subExpsRes . pure) . letSubExp "tan" <=< toExp $
          f
            (LeafExp (paramName x_tan_p) t)
            (LeafExp (paramName y_tan_p) t)
      pure $ Op $ Screma w [x_tan, y_tan] (mapSOAC lam)

basicFwd :: Pat Type -> StmAux () -> BasicOp -> ADM ()
basicFwd pat aux op = do
  pat_tan <- newTanPat pat
  case op of
    SubExp se -> do
      se_tan <- tangent se
      addStm $ Let pat_tan aux $ BasicOp $ SubExp se_tan
    Opaque opaqueop se -> do
      se_tan <- tangent se
      addStm $ Let pat_tan aux $ BasicOp $ Opaque opaqueop se_tan
    ArrayLit ses t -> do
      ses_tan <- mapM tangent ses
      addStm $ Let pat_tan aux $ BasicOp $ ArrayLit ses_tan t
    UnOp unop x -> do
      let t = unOpType unop
          x_pe = primExpFromSubExp t x
          dx = pdUnOp unop x_pe
      auxing aux $ letBindNames (patNames pat_tan) <=< withTan x $ \x_tan ->
        toExp $ primExpFromSubExp t x_tan ~*~ dx
    BinOp bop x y -> do
      let t = binOpType bop
      auxing aux . letBindNames (patNames pat_tan) <=< withTans t x y $
        \x_tan y_tan ->
          let (wrt_x, wrt_y) =
                pdBinOp bop (primExpFromSubExp t x) (primExpFromSubExp t y)
           in x_tan ~*~ wrt_x ~+~ y_tan ~*~ wrt_y
    CmpOp {} ->
      addStm $ Let pat_tan aux $ BasicOp op
    ConvOp cop x -> do
      x_tan <- tangent x
      addStm $ Let pat_tan aux $ BasicOp $ ConvOp cop x_tan
    Assert {} -> pure ()
    Index arr slice -> do
      arr_tan <- tangent arr
      dims <- shapeDims <$> askShape
      let slice' = Slice $ map sliceDim dims <> unSlice slice
      addStm $ Let pat_tan aux $ BasicOp $ Index arr_tan slice'
    Update safety arr slice se -> do
      arr_tan <- tangent arr
      se_tan <- tangent se
      addStm $ Let pat_tan aux $ BasicOp $ Update safety arr_tan slice se_tan
    Concat d (arr :| arrs) w -> do
      arr_tan <- tangent arr
      arrs_tans <- mapM tangent arrs
      r <- shapeRank <$> askShape
      addStm $ Let pat_tan aux $ BasicOp $ Concat (d + r) (arr_tan :| arrs_tans) w
    Manifest ds arr -> do
      arr_tan <- tangent arr
      r <- shapeRank <$> askShape
      addStm . Let pat_tan aux . BasicOp $
        Manifest ([0 .. r - 1] ++ map (+ r) ds) arr_tan
    Iota n _ _ it -> do
      shape <- askShape
      addStm . Let pat_tan aux . BasicOp $
        Replicate (shape <> Shape [n]) (intConst it 0)
    Replicate n x ->
      auxing aux $ letBind pat_tan <=< withTan x $ \x_tan ->
        pure $ BasicOp $ Replicate n x_tan
    Scratch t shape -> do
      tan_shape <- askShape
      addStm $ Let pat_tan aux $ BasicOp $ Scratch t $ shapeDims tan_shape <> shape
    Reshape k reshape arr -> do
      arr_tan <- tangent arr
      shape <- askShape
      addStm $ Let pat_tan aux $ BasicOp $ Reshape k (shape <> reshape) arr_tan
    Rearrange perm arr -> do
      arr_tan <- tangent arr
      r <- shapeRank <$> askShape
      addStm . Let pat_tan aux . BasicOp $
        Rearrange ([0 .. r - 1] <> map (+ r) perm) arr_tan
    _ -> error $ "basicFwd: Unsupported op " ++ prettyString op

fwdLambda :: Lambda SOACS -> ADM (Lambda SOACS)
fwdLambda (Lambda params ret body) = do
  params' <- bundleNewList params
  Lambda params'
    <$> bundleTangents ret
    <*> localScope (scopeOfLParams params') (fwdBody body)

fwdStreamLambda :: Lambda SOACS -> ADM (Lambda SOACS)
fwdStreamLambda (Lambda params ret body) = do
  params' <- (take 1 params ++) <$> bundleNewList (drop 1 params)
  Lambda params'
    <$> bundleTangents ret
    <*> localScope (scopeOfLParams params') (fwdBody body)

interleave :: [a] -> [a] -> [a]
interleave xs ys = concat $ transpose [xs, ys]

zeroFromSubExp :: SubExp -> ADM VName
zeroFromSubExp (Constant c) =
  letExp "zero" . BasicOp . SubExp . Constant $
    blankPrimValue (primValueType c)
zeroFromSubExp (Var v) = do
  t <- lookupType v
  letExp "zero" $ zeroExp t

fwdSOAC :: Pat Type -> StmAux () -> SOAC SOACS -> ADM ()
fwdSOAC pat aux (Screma size xs (ScremaForm f scs reds)) = do
  pat' <- bundleNewPat pat
  xs' <- bundleTangents xs
  f' <- fwdLambda f
  scs' <- mapM fwdScan scs
  reds' <- mapM fwdRed reds
  addStm $ Let pat' aux $ Op $ Screma size xs' $ ScremaForm f' scs' reds'
  where
    zeroTans lam =
      mapM (letSubExp "zero" . zeroExp <=< tanType) $ lambdaReturnType lam

    fwdScan :: Scan SOACS -> ADM (Scan SOACS)
    fwdScan sc = do
      op' <- fwdLambda $ scanLambda sc
      neutral_tans <- zeroTans $ scanLambda sc
      pure $
        Scan
          { scanNeutral = scanNeutral sc `interleave` neutral_tans,
            scanLambda = op'
          }
    fwdRed :: Reduce SOACS -> ADM (Reduce SOACS)
    fwdRed red = do
      op' <- fwdLambda $ redLambda red
      neutral_tans <- zeroTans $ redLambda red
      pure $
        Reduce
          { redComm = redComm red,
            redLambda = op',
            redNeutral = redNeutral red `interleave` neutral_tans
          }
fwdSOAC pat aux (Stream size xs nes lam) = do
  pat' <- bundleNewPat pat
  lam' <- fwdStreamLambda lam
  xs' <- bundleTangents xs
  nes_tan <- mapM (fmap Var . zeroFromSubExp) nes
  let nes' = interleave nes nes_tan
  addStm $ Let pat' aux $ Op $ Stream size xs' nes' lam'
fwdSOAC pat aux (Hist w arrs ops bucket_fun) = do
  pat' <- bundleNewPat pat
  ops' <- mapM fwdHist ops
  bucket_fun' <- fwdHistBucket bucket_fun
  arrs' <- bundleTangents arrs
  addStm $ Let pat' aux $ Op $ Hist w arrs' ops' bucket_fun'
  where
    n_indices = sum $ map (shapeRank . histShape) ops
    fwdBodyHist (Body _ stms res) = buildBody_ $ do
      mapM_ fwdStm stms
      let (res_is, res_vs) = splitAt n_indices res
      (res_is ++) <$> bundleTangents res_vs
    fwdHistBucket l@(Lambda params ret body) =
      let (r_is, r_vs) = splitAt n_indices ret
       in Lambda
            <$> bundleNewList params
            <*> ((r_is ++) <$> bundleTangents r_vs)
            <*> inScopeOf l (fwdBodyHist body)

    fwdHist :: HistOp SOACS -> ADM (HistOp SOACS)
    fwdHist (HistOp shape rf dest nes op) = do
      dest' <- bundleTangents dest
      nes_tan <- mapM (fmap Var . zeroFromSubExp) nes
      op' <- fwdLambda op
      pure $
        HistOp
          { histShape = shape,
            histRaceFactor = rf,
            histDest = dest',
            histNeutral = interleave nes nes_tan,
            histOp = op'
          }
fwdSOAC (Pat pes) aux (Scatter w ivs as lam) = do
  as_tan <- mapM (\(s, n, a) -> do a_tan <- tangent a; pure (s, n, a_tan)) as
  pes_tan <- mapM newTan pes
  ivs' <- bundleTangents ivs
  let (as_ws, as_ns, _as_vs) = unzip3 as
      n_indices = sum $ zipWith (*) as_ns $ map length as_ws
  lam' <- fwdScatterLambda n_indices lam
  let s = Let (Pat (pes ++ pes_tan)) aux $ Op $ Scatter w ivs' (as ++ as_tan) lam'
  addStm s
  where
    fwdScatterLambda :: Int -> Lambda SOACS -> ADM (Lambda SOACS)
    fwdScatterLambda n_indices (Lambda params ret body) = do
      params' <- bundleNewList params
      ret_tan <- mapM tangent $ drop n_indices ret
      body' <- fwdBodyScatter n_indices body
      let indices = concat $ replicate 2 $ take n_indices ret
          ret' = indices ++ drop n_indices ret ++ ret_tan
      pure $ Lambda params' ret' body'
    fwdBodyScatter :: Int -> Body SOACS -> ADM (Body SOACS)
    fwdBodyScatter n_indices (Body _ stms res) = do
      (res_tan, stms') <- collectStms $ do
        mapM_ fwdStm stms
        mapM tangent $ drop n_indices res
      let indices = concat $ replicate 2 $ take n_indices res
          res' = indices ++ drop n_indices res ++ res_tan
      pure $ mkBody stms' res'
fwdSOAC _ _ JVP {} =
  error "fwdSOAC: nested JVP not allowed."
fwdSOAC _ _ VJP {} =
  error "fwdSOAC: nested VJP not allowed."

fwdStm :: Stm SOACS -> ADM ()
fwdStm (Let pat aux (BasicOp (UpdateAcc safety acc i x))) = do
  pat' <- bundleNewPat pat
  x' <- bundleTangents x
  acc_tan <- tangent acc
  addStm $ Let pat' aux $ BasicOp $ UpdateAcc safety acc_tan i x'
fwdStm stm@(Let pat aux (BasicOp e)) = do
  -- XXX: this has to be too naive.
  unless (any isAcc $ patTypes pat) $ addStm stm
  basicFwd pat aux e
fwdStm stm@(Let pat _ (Apply f args _ _))
  | Just (ret, argts) <- M.lookup f builtInFunctions = do
      addStm stm
      arg_tans <-
        zipWith primExpFromSubExp argts <$> mapM (tangent . fst) args
      pat_tan <- newTanPat pat
      let arg_pes = zipWith primExpFromSubExp argts (map fst args)
      case pdBuiltin f arg_pes of
        Nothing ->
          error $ "No partial derivative defined for builtin function: " ++ prettyString f
        Just derivs -> do
          let convertTo tt e
                | e_t == tt = e
                | otherwise =
                    case (tt, e_t) of
                      (IntType tt', IntType ft) -> ConvOpExp (SExt ft tt') e
                      (FloatType tt', FloatType ft) -> ConvOpExp (FPConv ft tt') e
                      (Bool, FloatType ft) -> ConvOpExp (FToB ft) e
                      (FloatType tt', Bool) -> ConvOpExp (BToF tt') e
                      _ -> error $ "fwdStm.convertTo: " ++ prettyString (f, tt, e_t)
                where
                  e_t = primExpType e
          zipWithM_ (letBindNames . pure) (patNames pat_tan)
            =<< mapM toExp (zipWith (~*~) (map (convertTo ret) arg_tans) derivs)
fwdStm (Let pat aux (Match ses cases defbody (MatchDec ret ifsort))) = do
  cases' <- slocal' $ mapM (traverse fwdBody) cases
  defbody' <- slocal' $ fwdBody defbody
  pat' <- bundleNewPat pat
  ret' <- bundleTangents ret
  addStm $ Let pat' aux $ Match ses cases' defbody' $ MatchDec ret' ifsort
fwdStm (Let pat aux (Loop val_pats loop@(WhileLoop v) body)) = do
  val_pats' <- bundleNewList val_pats
  pat' <- bundleNewPat pat
  body' <-
    localScope (scopeOfFParams (map fst val_pats) <> scopeOfLoopForm loop) . slocal' $
      fwdBody body
  addStm $ Let pat' aux $ Loop val_pats' (WhileLoop v) body'
fwdStm (Let pat aux (Loop val_pats loop@(ForLoop i it bound) body)) = do
  pat' <- bundleNewPat pat
  val_pats' <- bundleNewList val_pats
  body' <-
    localScope (scopeOfFParams (map fst val_pats) <> scopeOfLoopForm loop) . slocal' $
      fwdBody body
  addStm $ Let pat' aux $ Loop val_pats' (ForLoop i it bound) body'
fwdStm (Let pat aux (WithAcc inputs lam)) = do
  inputs' <- forM inputs $ \(shape, arrs, op) -> do
    arrs_tan <- mapM tangent arrs
    op' <- case op of
      Nothing -> pure Nothing
      Just (op_lam, nes) -> do
        nes_tan <- mapM (fmap Var . zeroFromSubExp) nes
        op_lam' <- fwdLambda op_lam
        case op_lam' of
          Lambda ps ret body -> do
            let op_lam'' = Lambda (removeIndexTans (shapeRank shape) ps) ret body
            pure $ Just (op_lam'', interleave nes nes_tan)
    pure (shape, arrs <> arrs_tan, op')
  pat' <- bundleNewPat pat
  lam' <- fwdLambda lam
  addStm $ Let pat' aux $ WithAcc inputs' lam'
  where
    removeIndexTans 0 ps = ps
    removeIndexTans i (p : _ : ps) = p : removeIndexTans (i - 1) ps
    removeIndexTans _ ps = ps
fwdStm (Let pat aux (Op soac)) = fwdSOAC pat aux soac
fwdStm stm =
  error $ "unhandled forward mode AD for Stm: " ++ prettyString stm ++ "\n" ++ show stm

fwdBody :: Body SOACS -> ADM (Body SOACS)
fwdBody (Body _ stms res) = buildBody_ $ do
  mapM_ fwdStm stms
  bundleTangents res

fwdBodyTansLast :: Body SOACS -> ADM (Body SOACS)
fwdBodyTansLast (Body _ stms res) = buildBody_ $ do
  mapM_ fwdStm stms
  (res <>) <$> mapM tangent res

fwdJVP ::
  (MonadFreshNames m) =>
  Scope SOACS ->
  Shape ->
  Lambda SOACS ->
  m (Lambda SOACS)
fwdJVP scope shape (Lambda params _ body) =
  runADM shape . localScope scope $ do
    params_tan <- mapM newTan params
    mkLambda (params <> params_tan) $
      bodyBind =<< fwdBodyTansLast body
