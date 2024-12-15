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
import qualified DTS.UDTT as UD
import qualified System.IO as S
import Debug.Trace
import DTS.NeuralDTS.PreProcess (extractPredicateName, getTrainRelations, getTestRelations)
import DTS.NeuralDTS.Classifier.MLP (trainModel, testModel)
import Data.Map (mapMaybe)

inputsDir = "src/DTS/NeuralDTS/inputs"

processAndTrain :: Int -> Int -> [T.Text] -> IO ()
processAndTrain beam nbest posStr = do
  (posTrainRelations, negTrainRelations) <- getTrainRelations beam nbest posStr -- :: (Map.Map Int [([Int], Int)], Map.Map Int [([Int], Int)])
  putStrLn "Training Relations:"
  print posTrainRelations
  print negTrainRelations

  let modelName = "mlp-model"
  let posTrainRelations' = Map.map (map (\(xs, y) -> ((xs, y), 1.0))) posTrainRelations
  let negTrainRelations' = Map.map (map (\(xs, y) -> ((xs, y), 0.0))) negTrainRelations
  
  -- n項関係のモデルをトレーニング
  mapM_ (\(arity, relations) -> trainModel modelName relations arity) (Map.toList posTrainRelations')

processAndTest :: Int -> Int -> [T.Text] -> IO ()
processAndTest beam nbest str = do
  testRelationsByArity <- getTestRelations beam nbest str -- :: Map.Map Int [([Int], Int)]
  putStrLn "Test Relations:"
  print testRelationsByArity

  -- MLP モデルのロードとテスト
  let modelName = "mlp-model"
  mapM_ (\(arity, relations) -> testModel modelName relations arity) (Map.toList testRelationsByArity)

pretermsIndex :: Ord a => [a] -> [(a, Int)]
pretermsIndex xs = zip (L.nub xs) [0..]

-- CSVファイルを読み込む関数
readCsv :: FilePath -> IO [T.Text]
readCsv path = do
  content <- S.readFile path
  return $ T.lines (T.pack content)

testProcessAndTrain :: IO()
testProcessAndTrain = do
  let beam = 1
      nbest = 1

  -- CSVファイルを読み込む
  posStr <- readCsv (inputsDir ++ "/posStr.csv")
  
  -- テストデータを定義
  let testStr = [T.pack "次郎が踊る"]

  -- トレーニングとテストを実行
  processAndTrain beam nbest posStr
  processAndTest beam nbest testStr

-- testProcessAndTrain :: IO()
-- testProcessAndTrain = do
--   let beam = 1
--       nbest = 1
--       posStr = [ T.pack "太郎が走る"
--                , T.pack "太郎が踊る"
--                , T.pack "次郎が走る"
--                , T.pack "ジョンが吠える"
--                , T.pack "犬が走る"
--                , T.pack "猫が跳ぶ"
--                , T.pack "鳥が飛ぶ"
--                , T.pack "魚が泳ぐ"
--                , T.pack "車が走る"
--                ]
--       negStr = [ T.pack "ジョンが踊る"
--                , T.pack "太郎が吠える"
--                , T.pack "次郎が吠える"
--                , T.pack "花子が走る"
--                , T.pack "猫が泳ぐ"
--                , T.pack "鳥が跳ぶ"
--                , T.pack "魚が飛ぶ"
--                , T.pack "車が泳ぐ"
--                , T.pack "太郎が飛ぶ"
--                ]
--       testStr = [T.pack "次郎が踊る"]
--   processAndTrain beam nbest posStr negStr
--   processAndTest beam nbest testStr