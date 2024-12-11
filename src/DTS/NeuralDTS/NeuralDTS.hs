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

processAndTrain :: Int -> Int -> [T.Text] -> IO ()
processAndTrain beam nbest str = do
  trainRelations <- getTrainRelations beam nbest str -- :: [((Int, Int), Int)]
  putStrLn "Training Relations:"
  print trainRelations

  -- MLP モデルのトレーニング
  let modelName = "mlp-model"
  let posTrainRelations = addLabel trainRelations 1
  trainModel modelName posTrainRelations

processAndTest :: Int -> Int -> [T.Text] -> IO ()
processAndTest beam nbest str = do
  testRelations <- getTestRelations beam nbest str
  putStrLn "Test Relations:"
  print testRelations

  -- MLP モデルのロードとテスト
  let modelName = "mlp-model"
  testModel modelName testRelations

addLabel :: [((Int, Int), Int)] -> Int -> [(((Int, Int), Int), Float)]
addLabel dataList label = map (\x -> (x, fromIntegral label :: Float)) dataList

pretermsIndex :: Ord a => [a] -> [(a, Int)]
pretermsIndex xs = zip (L.nub xs) [0..]

testProcessAndTrain :: IO()
testProcessAndTrain = do
  let beam = 1
      nbest = 1
      str1 = [T.pack "太郎が走る", T.pack "太郎が歌う", T.pack "次郎が歩く", T.pack "次郎が走る"]
      str2 = [T.pack "次郎が歌う"]
  processAndTrain beam nbest str1
  processAndTest beam nbest str2