-- | Precise simplificaion rules for one pattern.
--   Examples include multiplication of Pow Symbols
--   and peeling off known Indexes from the beginning
--   or end of sum-of-slices.
module Futhark.Analysis.Proofs.AlgebraPC.UnaryRules
  ( simplifyPows,
    simplifyOneSum,
  )
where

import Control.Monad
import Data.Map.Strict qualified as M
import Data.Maybe
import Data.MultiSet qualified as MS
import Data.Set qualified as S
import Futhark.Analysis.Proofs.AlgebraPC.Symbol
import Futhark.SoP.SoP
import Futhark.SoP.Monad (MonadSoP, getEquivs)
import Futhark.SoP.FourierMotzkin qualified as FM

-----------------------------------------
--- 1. Simplifications related to Pow ---
-----------------------------------------

simplifyPows ::
  (MonadSoP Symbol e p m) =>
  (SoP Symbol  -> m (SoP Symbol)) -> SoP Symbol -> m (SoP Symbol)
simplifyPows simplifyLevel sop = do
  lst <- mapM simplifyTerm $ M.toList $ getTerms sop
  pure $ SoP $ M.fromList lst
  where
    -- simplifyTerm :: (Term Symbol, Integer) -> AlgM e (Term Symbol, Integer)
    simplifyTerm (Term mset, k) = do
      let (mset_pows, mset_others) = MS.partition hasPow mset
          mset_tup_pows = MS.mapMaybe mpowAsTup mset_pows
          lst_pows = map normalizePow $ MS.toOccurList mset_tup_pows
          (k', map_pows') = foldl combineSamePow (k, M.empty) lst_pows
      mset_pows'' <-
        forM (M.toList map_pows') $ \(b, p_sop) -> do
          p_sop' <- simplifyLevel p_sop
          -- \^ we simplify the exponents
          pure $ Pow (b, p_sop')
      pure $ (Term (MS.fromList mset_pows'' <> mset_others), k')
    --
    normalizePow :: ((Integer, SoP Symbol), Int) -> (Integer, SoP Symbol)
    normalizePow ((base, expnt), p) =
      (base, (int2SoP (fromIntegral p)) .*. expnt)
    mpowAsTup :: Symbol -> Maybe (Integer, SoP Symbol)
    mpowAsTup (Pow (base, expnt)) = Just (base, expnt)
    mpowAsTup _ = Nothing

combineSamePow ::
  (Integer, M.Map Integer (SoP Symbol)) ->
  (Integer, SoP Symbol) ->
  (Integer, M.Map Integer (SoP Symbol))
combineSamePow (q, tab) (b, sop) =
  let (q', sop') =
        case getPowOfFactor q b of
          (_, 0) -> (q, sop)
          (r, p) -> (r, int2SoP p .+. sop)
      sop'' = maybe sop' (.+. sop') $ M.lookup b tab
   in (q', M.insert b sop'' tab)
  where
    getPowOfFactor :: Integer -> Integer -> (Integer, Integer)
    getPowOfFactor qq bb = getPowOfFactorTR qq bb 0
    getPowOfFactorTR qq bb pr
      | qq `mod` bb /= 0 = (qq, pr)
    getPowOfFactorTR qq bb pr =
      getPowOfFactorTR (qq `div` bb) bb (pr + 1)

---------------------------------------------------------------
--- 2. Simplification of each (individual) slice sum:       ---
---    2.1. sum x[lb .. ub] => 0     whenever lb  > ub      ---
---    2.2. sum x[lb .. ub] => x[lb] whenever lb == ub      ---
---    2.3. peeling off first/last known elements of a sum  ---
---------------------------------------------------------------

simplifyOneSum ::
  (MonadSoP Symbol e p m) => SoP Symbol -> m (Bool, SoP Symbol)
simplifyOneSum sop = do
  equivs <- getEquivs
  sop' <- elimEmptySums sop
  let (succ1, sop'') = transfSum2Idx sop'
  let (succ2, sop''')= peelOffSumsFP equivs sop''
  pure (succ1 || succ2, sop''')

elimEmptySums :: 
  (MonadSoP Symbol e p m) => SoP Symbol -> m (SoP Symbol)
elimEmptySums sop = do
  sopFromList <$> (filterM predTerm $ sopToList sop)
  where
    emptySumSym (Sum _ lb ub) = lb FM.$>$ ub
    emptySumSym _ = pure False
    predTerm (Term ms, _) = do
      tmps <- mapM (emptySumSym . fst) $ MS.toOccurList ms
      pure $ all not tmps

transfSum2Idx :: SoP Symbol -> (Bool, SoP Symbol)
transfSum2Idx sop
  | tgt_sums <- filter isOneElmSum $ S.toList $ free sop,
    not (null tgt_sums) =
  let subs = M.fromList $ zip tgt_sums $ map sum2Idx tgt_sums
  in  (True, substitute subs sop)
  where
    isOneElmSum (Sum _ lb ub) = lb == ub
    isOneElmSum _ = False
    sum2Idx (Sum idxsym lb _) = Idx idxsym lb
    sum2Idx _ = error "Unreachable case reached in transfSum2Idx."
transfSum2Idx sop = (False, sop)

peelOffSumsFP :: M.Map Symbol (SoP Symbol) -> SoP Symbol -> (Bool, SoP Symbol)
peelOffSumsFP equivs sop
  | hasPeelableSums sop =
      case peelOffSums equivs sop of
        (False, _) -> (False, sop)
        (True, sop') ->
          -- fix point
          let (_, sop'') = peelOffSumsFP equivs sop'
          in  (True, sop'')
  where
    hasPeelableSums = any hasPeelableSumSym . S.toList . free
    hasPeelableSumSym (Sum nm beg end) =
      isJust (M.lookup (Idx nm beg) equivs)
        || isJust (M.lookup (Idx nm end) equivs)
    hasPeelableSumSym _ = False
--
peelOffSumsFP _ sop = (False, sop)

peelOffSums :: M.Map Symbol (SoP Symbol) -> SoP Symbol -> (Bool, SoP Symbol)
peelOffSums equivs sop = do
  case foldl peelTerm Nothing (M.toList (getTerms sop)) of
    Nothing -> (False, sop)
    Just (old_term_sop, new_sop) ->
      (True, (sop .-. old_term_sop) .+. new_sop)
  where
    peelTerm acc@(Just {}) _ = acc
    peelTerm Nothing (t, k) =
      let mres = foldl peelSymb Nothing $ MS.toOccurList $ getTerm t
       in case mres of
            Nothing -> Nothing
            Just (sop_sym, sum_sym) ->
              let ms' = MS.delete sum_sym $ getTerm t
                  sop' = sop_sym .*. term2SoP (Term ms') k
               in Just (term2SoP t k, sop')
    peelSymb acc@(Just {}) _ = acc
    peelSymb Nothing (sym@(Sum nm beg end), 1) =
      -- \^ ToDo: extend for any multiplicity >= 1
      let mfst_el = M.lookup (Idx nm beg) equivs
          mlst_el = M.lookup (Idx nm end) equivs
       in case (mfst_el, mlst_el) of
            (Just fst_el, Nothing) ->
              let new_sum = Sum nm (beg .+. int2SoP 1) end
               in Just (fst_el .+. sym2SoP new_sum, sym)
            (Nothing, Just lst_el) ->
              let new_sum = Sum nm beg (end .-. int2SoP 1)
               in Just (lst_el .+. sym2SoP new_sum, sym)
            (Just fst_el, Just lst_el) ->
              let new_sum = Sum nm (beg .+. int2SoP 1) (end .-. int2SoP 1)
               in Just (fst_el .+. lst_el .+. sym2SoP new_sum, sym)
            (Nothing, Nothing) -> Nothing
    peelSymb Nothing _ = Nothing
