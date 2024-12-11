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

processAndTrain :: Int -> Int -> [T.Text] -> [T.Text] -> IO ()
processAndTrain beam nbest posStr negStr = do
  (posTrainRelations, negTrainRelations) <- getTrainRelations beam nbest posStr negStr -- :: [((Int, Int), Int)]
  putStrLn "Training Relations:"
  print posTrainRelations
  print negTrainRelations

  -- MLP モデルのトレーニング
  let modelName = "mlp-model"
  let posTrainRelations' = addLabel posTrainRelations 1
  let negTrainRelations' = addLabel negTrainRelations 0
  trainModel modelName (posTrainRelations' ++ negTrainRelations')

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
      posStr = [T.pack "太郎が走る", T.pack "太郎が踊る", T.pack "次郎が走る", T.pack "犬が吠える"]
      negStr = [T.pack "犬が踊る", T.pack "太郎が吠える", T.pack "次郎が吠える"]
      testStr = [T.pack "次郎が踊る"]
  processAndTrain beam nbest posStr negStr
  processAndTest beam nbest testStr