{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE InstanceSigs #-}

module DTS.NeuralDTS.Classifier.Utils (
    getLineCount,
    splitIntoFolds,
    splitAtFold,
    crossValidation
) where

import System.Random (newStdGen)
import System.Random.Shuffle (shuffle')
import Data.List (unfoldr)
import qualified System.IO as S
import Torch
import DTS.NeuralDTS.Classifier

crossValidation :: (Classifier n, Randomizable spec n) => Int -> spec -> (String -> spec -> [(([Int], Int), Float)] -> Int -> IO ()) -> (String -> spec -> [(([Int], Int), Float)] -> Int -> IO Double) -> [(([Int], Int), Float)] -> [(([Int], Int), Float)] -> Int -> IO Double
crossValidation k spec trainModel testModel dataSet addData arity = do
  putStrLn $ "Cross Validation with " ++ show k ++ " folds"
  S.hFlush S.stdout
  -- データをシャッフル
  gen <- newStdGen
  let shuffledData = shuffle' dataSet (length dataSet) gen
  -- データをk分割
  let folds = splitIntoFolds k shuffledData
  -- 各フォールドで訓練と検証を行う
  accuracies <- mapM (\(i, fold) -> do
        let (trainFolds, validFold) = splitAtFold i folds
        putStrLn $ "Fold " ++ show i ++ ": Train size = " ++ show (length (concat trainFolds)) ++ ", Valid size = " ++ show (length validFold)
        let trainData' = concat trainFolds ++ addData
        gen1 <- newStdGen
        let shuffledTrainData = shuffle' trainData' (length trainData') gen1
        let validData' = validFold
        gen2 <- newStdGen
        let shuffledValidData = shuffle' validData' (length validData') gen2
        putStrLn $ "Fold " ++ show i ++ ": AddTrain size = " ++ show (length shuffledTrainData) ++ ", Valid size = " ++ show (length shuffledValidData)
        let modelName = "model_fold_" ++ show i
        _ <- trainModel modelName spec shuffledTrainData arity
        accuracy <- testModel modelName spec shuffledValidData arity
        return accuracy
        ) (zip [0..k-1] folds)
  let averageAccuracy = sum accuracies / fromIntegral k
  putStrLn $ "Cross Validation finished. Average accuracy: " ++ show averageAccuracy ++ "%"
  return averageAccuracy

getLineCount :: FilePath -> IO Int
getLineCount path = do
  content <- readFile path
  return $ length (lines content)

splitIntoFolds :: Int -> [a] -> [[a]]
splitIntoFolds k xs = let foldSize = length xs `Prelude.div` k
                          remainder = length xs `mod` k
                      in Prelude.take k $ unfoldr (\b -> if null b then Nothing else Just (splitAt (foldSize + if remainder > 0 then 1 else 0) b)) xs

splitAtFold :: Int -> [[a]] -> ([[a]], [a])
splitAtFold i folds = let (before, after) = splitAt i folds
                      in if null after
                         then error "splitAtFold: after is empty"
                         else (before ++ tail after, head after)