{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE InstanceSigs #-}

module DTS.NeuralDTS.Classifier.MLP (
  trainModel
  ,testModel
  ,MLPSpec(..)
  ) where

import Control.Monad (when)
import Data.List (transpose)
import qualified Data.Binary as B
import GHC.Generics ( Generic )
import ML.Exp.Chart (drawLearningCurve, drawConfusionMatrix)
import ML.Exp.Classification (showClassificationReport)
import Torch
import Torch.Control (mapAccumM, makeBatch)
import qualified Torch.Train
import System.Directory (createDirectoryIfMissing)
import System.FilePath ((</>))
import System.Mem (performGC)
import qualified System.IO as S
import System.IO.Unsafe (unsafePerformIO)
import System.IO (withFile, IOMode(..), hFileSize)
import System.Random.Shuffle (shuffle')

import DTS.NeuralDTS.Classifier

modelsDir = "src/DTS/NeuralDTS/models"
dataDir = "src/DTS/NeuralDTS/dataSet"
imagesDir = "src/DTS/NeuralDTS/images"
indexNum = 36

data MLPSpec = MLPSpec
  { 
    entity_num_embed :: Int,
    relation_num_embed :: Int,
    entity_features :: Int,
    relation_features :: Int,
    hidden_dim1 :: Int,
    hidden_dim2 :: Int,
    output_feature :: Int,
    arity :: Int
  } deriving (Generic, B.Binary, Show)

data MLP = MLP
  { 
    entity_emb :: Parameter,
    relation_emb :: Parameter,
    linear_layer1 :: Linear,
    linear_layer2 :: Linear,
    linear_layer3 :: Linear
  } deriving (Generic, Parameterized)

instance Show MLP where
  show MLP {..} = show entity_emb ++ "\n" ++ show relation_emb ++ "\n" ++ show linear_layer1  ++ "\n"++ show linear_layer2 ++ "\n"++ show linear_layer3 ++ "\n"

instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} = MLP
    <$> (makeIndependent =<< randnIO' [entity_num_embed, entity_features])
    <*> (makeIndependent =<< randnIO' [relation_num_embed, relation_features])
    <*> sample (LinearSpec (entity_features * arity + relation_features) hidden_dim1) -- 最初の線形層の入力サイズ 160
    <*> sample (LinearSpec hidden_dim1 hidden_dim2)
    <*> sample (LinearSpec hidden_dim2 output_feature)

instance Classifier MLP where
  classify :: MLP -> RuntimeMode -> Tensor -> [Tensor] -> Tensor
  classify MLP {..} _ predTensor entitiesTensor =
    let pred2 = embedding' (toDependent relation_emb) predTensor
        entities = map (embedding' (toDependent entity_emb)) entitiesTensor
        input = cat (Dim 1) (pred2 : entities)
        -- input = cat (Dim 1) (entities ++ [relation])
        nonlinearity = Torch.sigmoid
    in nonlinearity $ linear linear_layer3 $ nonlinearity $ linear linear_layer2 $ nonlinearity 
       $ linear linear_layer1 $ input

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

batchSize :: Int
batchSize = 256
numIters :: Integer
numIters = 10000
myDevice :: Device
myDevice = Device CUDA 0
-- myDevice = Device CPU 0
mode :: RuntimeMode
mode = Train
lr :: LearningRate
lr = 5e-2

calculateLoss :: (Classifier n) => n -> [(([Int], Int), Float)] -> IO Tensor
calculateLoss model dataSet = do
  let (input, label) = unzip dataSet
      (entityIdxes, relationIdxes) = unzip input :: ([[Int]], [Int])
  let maxLength = maximum (map length entityIdxes)
  let entityIdxesPadded = map (\lst -> Prelude.take maxLength (lst ++ Prelude.repeat 0)) entityIdxes
  let entityIdxTensors = map (toDevice myDevice . asTensor) (Data.List.transpose entityIdxesPadded)
  let relationIdxTensor = toDevice myDevice $ asTensor (relationIdxes :: [Int])
  let teachers = toDevice myDevice $ asTensor (label :: [Float]) :: Tensor
  let prediction = squeezeAll $ classify model mode relationIdxTensor entityIdxTensors
  return $ binaryCrossEntropyLoss' teachers prediction

trainModel :: String -> MLPSpec -> [(([Int], Int), Float)] -> Int -> IO ()
trainModel modelName spec trainData arity = do
  putStrLn $ "trainModel : " ++ show arity
  S.hFlush S.stdout

  -- データが空でないことを確認
  when (null trainData) $ do
    error "Training data is empty. Check your input data."

  -- 設定値を出力
  putStrLn $ "Model Specification: " ++ show spec
  putStrLn $ "Batch Size: " ++ show batchSize
  putStrLn $ "Number of Iterations: " ++ show numIters
  putStrLn $ "Device: " ++ show myDevice
  putStrLn $ "Runtime Mode: " ++ show mode
  putStrLn $ "Learning Rate: " ++ show lr

  -- 学習プロセス
  let batchedTrainSet = makeBatch batchSize trainData

  -- model
  initModel <- toDevice myDevice <$> sample spec
  let optimizer = GD
  -- let optimizer = mkAdam 0 0.9 0.999 (flattenParameters initModel)
  
  ((trainedModel, _), losses) <- mapAccumM [1..numIters] (initModel, optimizer) $ \epoch (model', opt') -> do
    (batchTrained@(batchModel, _), batchLosses) <- mapAccumM batchedTrainSet (model', opt') $ 
      \batch (model, opt) -> do
        loss <- calculateLoss model batch
        updated <- runStep model opt loss lr
        return (updated, asValue loss::Float)
    -- batch の長さでlossをわる
    let batchloss = sum batchLosses / (fromIntegral (length batchLosses)::Float)
    -- when (epoch `mod` 10 == 0) $ do
    --   putStrLn $ "Iteration: " ++ show epoch ++ " | Loss: " ++ show batchloss
    putStrLn $ "Iteration: " ++ show epoch ++ " | Loss: " ++ show batchloss
    S.hFlush S.stdout
    return (batchTrained, batchloss)

  -- モデルを保存
  let modelDir = modelsDir </> show indexNum
  createDirectoryIfMissing True modelDir
  Torch.Train.saveParams trainedModel (modelDir </> modelName ++ "_arity" ++ show arity ++ ".model")
  B.encodeFile (modelDir </> modelName ++ "_arity" ++ show arity ++ ".model-spec") spec
  putStrLn $ "Model saved to " ++ modelDir ++ "/" ++ modelName ++ "_arity" ++ show arity ++ ".model"
  S.hFlush S.stdout

  -- 学習曲線
  let imageDir = imagesDir </> show indexNum
  createDirectoryIfMissing True imageDir
  let imagePath = imageDir </> modelName ++ "_learning-curve-training_" ++ show arity ++ ".png"
  drawLearningCurve imagePath "Learning Curve" [("", reverse losses)]
  putStrLn $ "drawLearningCurve to " ++ imagePath

testModel :: String -> MLPSpec -> [(([Int], Int), Float)] -> Int -> IO (Double, Double, Double, Double)
testModel modelName spec testRelations arity = do
  putStrLn "testModel"

  let modelDir = modelsDir </> show indexNum
  loadedModel <- Torch.Train.loadParams spec (modelDir </> modelName ++ "_arity" ++ show arity ++ ".model")
  putStrLn $ "Model loaded from " ++ modelDir ++ "/" ++ modelName ++ "_arity" ++ show arity ++ ".model"

  putStrLn "Testing relations:"
  results <- mapM (\((entities, p), label) -> do
                        let entityTensors = map (\e -> toDevice myDevice $ asTensor ([fromIntegral e :: Int] :: [Int])) entities
                        let rTensor = toDevice myDevice $ asTensor ([fromIntegral p :: Int] :: [Int])
                        let output = classify loadedModel Eval rTensor entityTensors
                        let confidence = asValue (fst (maxDim (Dim 1) RemoveDim output)) :: Float
                        let confThreshold = 0.5
                        let prediction = if confidence >= confThreshold then 1 else 0 :: Int
                        return (label, fromIntegral prediction :: Float))
                    testRelations

  -- 精度、再現率、F1スコアの計算
  let (tp, fp, fn, tn) = foldl (\(tp, fp, fn, tn) (label, prediction) ->
                                  case (label, prediction) of
                                    (1.0, 1.0) -> (tp + 1, fp, fn, tn)
                                    (1.0, 0.0) -> (tp, fp, fn + 1, tn)
                                    (0.0, 1.0) -> (tp, fp + 1, fn, tn)
                                    (0.0, 0.0) -> (tp, fp, fn, tn + 1)
                                    _ -> (tp, fp, fn, tn)
                                ) (0, 0, 0, 0) results

  let precision = if tp + fp == 0 then 0 else fromIntegral tp / fromIntegral (tp + fp)
  let recall = if tp + fn == 0 then 0 else fromIntegral tp / fromIntegral (tp + fn)
  let f1Score = if precision + recall == 0 then 0 else 2 * (precision * recall) / (precision + recall)

  let correctPredictions = tp + tn
  let totalPredictions = length results
  let accuracy = (fromIntegral correctPredictions / fromIntegral totalPredictions) * 100 :: Double

  putStrLn $ "Accuracy: " ++ show accuracy ++ "%"
  putStrLn $ "Precision: " ++ show precision
  putStrLn $ "Recall: " ++ show recall
  putStrLn $ "F1 Score: " ++ show f1Score

  -- -- 分類レポートの生成
  -- let imageDir = imagesDir </> show indexNum
  -- let classificationReport = showClassificationReport 2 (zip (map snd results) (map fst results))
  -- T.putStr classificationReport
  -- let classificationReportFileName = imageDir </> modelName ++ "_classification_report_" ++ show arity ++ ".txt"
  -- B.writeFile classificationReportFileName (E.encodeUtf8 classificationReport)

  -- -- 混同行列の描画
  -- let confusionMatrixFileName = imageDir </> modelName ++ "_confusion_matrix_" ++ show arity ++ ".png"
  -- drawConfusionMatrix confusionMatrixFileName 2 (zip (map snd results) (map fst results))
  -- putStrLn $ "Confusion matrix saved to " ++ confusionMatrixFileName

  return (accuracy, precision, recall, f1Score)