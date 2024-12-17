{-# LANGUAGE DeriveAnyClass #-}
-- {-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module DTS.NeuralDTS.NeuralDTS  (
  processAndTrain,
  testProcessAndTrain
  ) where

import qualified Data.Text.Lazy as T      --text
import qualified Data.Text.Lazy.IO as T   --text
import qualified Data.List as L           --base
import qualified Data.Map as Map
import qualified Interface.Text as T
import qualified System.IO as S
import Debug.Trace
import DTS.NeuralDTS.PreProcess (extractPredicateName, getTrainRelations, getTestRelations)
import DTS.NeuralDTS.Classifier.MLP (trainModel, testModel)
import qualified Parser.ChartParser as CP
import Parser.Language (jpOptions) 
import qualified Parser.Language.Japanese.Juman.CallJuman as Juman
import qualified Parser.Language.Japanese.Lexicon as L (LexicalResource(..), lexicalResourceBuilder, LexicalItems, lookupLexicon, setupLexicon, emptyCategories, myLexicon)

inputsDir = "src/DTS/NeuralDTS/inputs"

processAndTrain :: CP.ParseSetting -> [T.Text] -> IO ()
processAndTrain ps posStr = do
  (posTrainRelations, negTrainRelations) <- getTrainRelations ps posStr -- :: (Map.Map Int [([Int], Int)], Map.Map Int [([Int], Int)])
  putStrLn "Training Relations:"
  print posTrainRelations
  print negTrainRelations

  let modelName = "mlp-model"
  let posTrainRelations' = Map.map (map (\(xs, y) -> ((xs, y), 1.0))) posTrainRelations
  let negTrainRelations' = Map.map (map (\(xs, y) -> ((xs, y), 0.0))) negTrainRelations
  let allTrainRelations = Map.unionWith (++) posTrainRelations' negTrainRelations'
  
  -- n項関係のモデルをトレーニング
  mapM_ (\(arity, relations) -> trainModel modelName relations arity) (Map.toList allTrainRelations)

processAndTest :: CP.ParseSetting -> [T.Text] -> IO ()
processAndTest ps str = do
  testRelationsByArity <- getTestRelations ps str -- :: Map.Map Int [([Int], Int)]
  putStrLn "Test Relations:"
  print testRelationsByArity

  -- MLP モデルのロードとテスト
  let modelName = "mlp-model"
  mapM_ (\(arity, relations) -> testModel modelName relations arity) (Map.toList testRelationsByArity)

-- CSVファイルを読み込む関数
readCsv :: FilePath -> IO [T.Text]
readCsv path = do
  content <- S.readFile path
  return $ T.lines (T.pack content)

testProcessAndTrain :: IO()
testProcessAndTrain = do
  -- CSVファイルを読み込む
  posStr <- readCsv (inputsDir ++ "/test.csv")
  
  -- テストデータを定義
  let testStr = [T.pack "次郎が踊る"]

  lr <- L.lexicalResourceBuilder Juman.KWJA
  let ps = CP.ParseSetting jpOptions lr 1 1 1 1 True Nothing Nothing True False

  -- トレーニングとテストを実行
  processAndTrain ps posStr
  processAndTest ps testStr