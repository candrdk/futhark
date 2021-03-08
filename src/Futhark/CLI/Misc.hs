{-# LANGUAGE FlexibleContexts #-}

-- | Various small subcommands that are too simple to deserve their own file.
module Futhark.CLI.Misc
  ( mainImports,
    mainHash,
    mainDataget,
  )
where

import Control.Monad.State
import qualified Data.ByteString.Lazy as BS
import Data.Function (on)
import Data.Hashable (hash)
import Data.List (isInfixOf, isPrefixOf, nubBy)
import qualified Data.Text.IO as T
import Futhark.Compiler
import Futhark.Test
import Futhark.Util (hashIntText)
import Futhark.Util.Options
import Futhark.Util.Pretty (prettyText)
import System.Environment (getExecutablePath)
import System.Exit
import System.FilePath
import System.IO

-- | @futhark imports@
mainImports :: String -> [String] -> IO ()
mainImports = mainWithOptions () [] "program" $ \args () ->
  case args of
    [file] -> Just $ do
      (_, prog_imports, _) <- readProgramOrDie file
      liftIO $
        putStr $
          unlines $
            map (++ ".fut") $
              filter (\f -> not ("prelude/" `isPrefixOf` f)) $
                map fst prog_imports
    _ -> Nothing

-- | @futhark hash@
mainHash :: String -> [String] -> IO ()
mainHash = mainWithOptions () [] "program" $ \args () ->
  case args of
    [file] -> Just $ do
      prog <- readUntypedProgramOrDie file
      liftIO $ T.putStrLn $ hashIntText $ hash $ prettyText prog
    _ -> Nothing

-- | @futhark dataget@
mainDataget :: String -> [String] -> IO ()
mainDataget = mainWithOptions () [] "program dataset" $ \args () ->
  case args of
    [file, dataset] -> Just $ dataget file dataset
    _ -> Nothing
  where
    dataget prog dataset = do
      let dir = takeDirectory prog

      runs <- testSpecRuns <$> testSpecFromFileOrDie prog

      let exact = filter ((dataset ==) . runDescription) runs
          infixes = filter ((dataset `isInfixOf`) . runDescription) runs

      futhark <- FutharkExe <$> getExecutablePath

      case nubBy ((==) `on` runDescription) $
        if null exact then infixes else exact of
        [x] -> BS.putStr =<< getValuesBS futhark dir (runInput x)
        [] -> do
          hPutStr stderr $ "No dataset '" ++ dataset ++ "'.\n"
          hPutStr stderr "Available datasets:\n"
          mapM_ (hPutStrLn stderr . ("  " ++) . runDescription) runs
          exitFailure
        runs' -> do
          hPutStr stderr $ "Dataset '" ++ dataset ++ "' ambiguous:\n"
          mapM_ (hPutStrLn stderr . ("  " ++) . runDescription) runs'
          exitFailure

    testSpecRuns = testActionRuns . testAction
    testActionRuns CompileTimeFailure {} = []
    testActionRuns (RunCases ios _ _) = concatMap iosTestRuns ios
