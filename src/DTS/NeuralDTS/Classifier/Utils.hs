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

crossValidation :: (Classifier n, Randomizable spec n) => Int -> spec -> 
  (String -> spec -> [(([Int], Int), Float)] -> Int -> IO ()) -> 
  (String -> spec -> [(([Int], Int), Float)] -> Int -> IO Double) -> 
  [(([Int], Int), Float)] -> [(([Int], Int), Float)] -> [(([Int], Int), Float)] -> [(([Int], Int), Float)] -> Int -> IO Double
crossValidation k spec trainModel testModel posOrgData posAddData negOrgData negAddData arity = do
  putStrLn $ "Cross Validation with " ++ show k ++ " folds"
  S.hFlush S.stdout
  -- データをk分割
  let posFolds = splitIntoFolds k posOrgData
  let negFolds = splitIntoFolds k negOrgData
  -- 各フォールドで訓練と検証を行う
  accuracies <- mapM (\(i) -> do
        let (trainPosFolds, validPosFold) = splitAtFold i posFolds
            (trainNegFolds, validNegFold) = splitAtFold i negFolds
        let trainFolds = trainPosFolds ++ trainNegFolds
        let validFold = validPosFold ++ validNegFold
        putStrLn $ "Fold " ++ show i ++ ": OrgTrain size = " ++ show (length (concat trainFolds)) ++ ", Valid size = " ++ show (length validFold)
        let addedTrainData = concat trainFolds ++ posAddData ++ negAddData
        gen1 <- newStdGen
        let shuffledTrainData = shuffle' addedTrainData (length addedTrainData) gen1
        gen2 <- newStdGen
        let shuffledValidData = shuffle' validFold (length validFold) gen2
        putStrLn $ "Fold " ++ show i ++ ": AddTrain size = " ++ show (length shuffledTrainData) ++ ", Valid size = " ++ show (length shuffledValidData)
        let modelName = "model_fold_" ++ show i
        _ <- trainModel modelName spec shuffledTrainData arity
        accuracy <- testModel modelName spec shuffledValidData arity
        return accuracy
        ) [0..k-1]
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