{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE InstanceSigs #-}

module DTS.NeuralDTS.Classifier.NTN (
    trainModel
    ,testModel
    ,NTNSpec(..)
) where

import Control.Monad (when, replicateM)
import Data.Binary as B
import Data.List (foldl', intersperse, scanl', transpose, unfoldr)
import Debug.Trace
import GHC.Generics
import System.FilePath ((</>))
import qualified System.IO as S
import System.IO.Unsafe (unsafePerformIO)
import System.Directory (createDirectoryIfMissing)
import ML.Exp.Chart (drawLearningCurve)

import Torch
import Torch.Control (mapAccumM, makeBatch)
import Torch.Functional.Internal (mm, bmm)
import Torch.Layer.Linear (LinearHypParams(..), LinearParams, linearLayer)
import qualified Torch.Train

import DTS.NeuralDTS.Classifier (Classifier(..))
import DTS.NeuralDTS.Classifier.Utils as Utils ( getLineCount )

modelsDir = "src/DTS/NeuralDTS/models"
dataDir = "src/DTS/NeuralDTS/dataSet"
imagesDir = "src/DTS/NeuralDTS/images"
indexNum = 12

data NTNSpec = NTNSpec
  { entity_num_embed :: Int,
    relation_num_embed :: Int,
    embedding_features :: Int,
    tensor_dim :: Int,
    num_arguments :: Int,
    dropout_probability :: Double
  } deriving (Generic, Binary, Show)

data NTN = NTN
  { 
    entity_emb :: Parameter,
    relation_emb :: Parameter,
    tensor_layers :: [Parameter],
    linear_layers :: [LinearParams],
    final_linear :: LinearParams,
    dropout_prob :: Double
  } deriving (Generic, Parameterized)

instance Randomizable NTNSpec NTN where
  sample NTNSpec {..} = do
    tensors <- replicateM num_arguments $ makeIndependent =<< randnIO' [embedding_features, embedding_features, tensor_dim]
    linears <- replicateM num_arguments $ sample (LinearHypParams myDevice True (embedding_features * 2) tensor_dim)
    NTN
      <$> (makeIndependent =<< randnIO' [entity_num_embed, embedding_features])
      <*> (makeIndependent =<< randnIO' [relation_num_embed, embedding_features])
      <*> pure tensors
      <*> pure linears
      <*> sample (LinearHypParams myDevice True tensor_dim 1)
      <*> pure dropout_probability

instance Classifier NTN where
  classify :: NTN -> RuntimeMode -> Tensor -> [Tensor] -> Tensor
  classify NTN {..} mode relation_idx entity_idxes =
    let entities = map (\idx -> embedding' (toDependent entity_emb) idx) entity_idxes
        relation = embedding' (toDependent relation_emb) relation_idx
        tensor_results = zipWith (tensorLinear relation) entities (zip linear_layers tensor_layers)
        aggregated_result = foldl1 (+) tensor_results
        -- output = Torch.sigmoid $ linearLayer final_linear $ Torch.tanh aggregated_result
        dropped_out = unsafePerformIO . dropout dropout_prob (mode == Train) $ aggregated_result
        output = Torch.sigmoid $ linearLayer final_linear $ Torch.tanh dropped_out
    in output

tensorLinear :: Tensor -> Tensor -> (LinearParams, Parameter) -> Tensor
tensorLinear relation entity (linear_layer, tensor_layer) =
  let d = shape entity !! 1 -- Embedding size
      batchSize = shape entity !! 0
      -- entityベクトルとrekationベクトルの連結
      entity_relation = cat (Dim 1) [entity, relation]
      -- 線形変換
      linear_product = linearLayer linear_layer entity_relation
      -- テンソル変換
      entity_tensor = mm entity (view [d, -1] (toDependent tensor_layer)) -- (batchSize, d) * (d, tensor_dim)
      -- bmm用にtensor_productの次元を調整
      entity_tensor_reshaped = view [batchSize, d, -1] entity_tensor
      relation_reshaped = view [batchSize, -1, 1] relation -- (batchSize, tensor_dim, 1)
      tensor_product = bmm entity_tensor_reshaped relation_reshaped -- (batchSize, tensor_dim, 1)
      tensor_product_shaped = reshape [batchSize, shape tensor_product !! 1] tensor_product -- (batchSize, tensor_dim)
  in tensor_product_shaped + linear_product

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

batchSize :: Int
batchSize = 256
numIters :: Integer
numIters = 100
myDevice :: Device
myDevice = Device CUDA 0
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

trainModel :: String -> NTNSpec -> [(([Int], Int), Float)] -> Int -> IO ()
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
--   let optimizer = GD
  let optimizer = mkAdam 0 0.9 0.999 (flattenParameters initModel)
  
  ((trainedModel, _), losses) <- mapAccumM [1..numIters] (initModel, optimizer) $ \epoch (model', opt') -> do
    (batchTrained@(batchModel, _), batchLosses) <- mapAccumM batchedTrainSet (model', opt') $ 
      \batch (model, opt) -> do
        loss <- calculateLoss model batch
        updated <- runStep model opt loss lr
        return (updated, asValue loss::Float)
    -- batch の長さでlossをわる
    let batchloss = sum batchLosses / (fromIntegral (length batchLosses)::Float)
    putStrLn $ "Iteration: " ++ show epoch ++ " | Loss: " ++ show batchloss
    S.hFlush S.stdout
    return (batchTrained, batchloss)

  -- モデルを保存
  let modelDir = modelsDir </> show indexNum
  createDirectoryIfMissing True modelDir
  Torch.saveParams trainedModel (modelDir </> modelName ++ "_arity" ++ show arity ++ ".model")
  B.encodeFile (modelDir </> modelName ++ "_arity" ++ show arity ++ ".model-spec") spec
  putStrLn $ "Model saved to " ++ modelDir ++ "/" ++ modelName ++ "_arity" ++ show arity ++ ".model"
  S.hFlush S.stdout

  -- 学習曲線
  let imageDir = imagesDir </> show indexNum
  createDirectoryIfMissing True imageDir
  let imagePath = imageDir </> modelName ++ "_learning-curve-training_" ++ show arity ++ ".png"
  drawLearningCurve imagePath "Learning Curve" [("", reverse losses)]
  putStrLn $ "drawLearningCurve to " ++ imagePath

testModel :: String -> NTNSpec -> [(([Int], Int), Float)] -> Int -> IO Double
testModel modelName spec testRelations arity = do
  putStrLn "testModel"

  loadedModel <- Torch.Train.loadParams spec (modelsDir </> show indexNum </> modelName ++ "_arity" ++ show arity ++ ".model")
  putStrLn $ "Model loaded from models/" ++ show indexNum ++ "/" ++ modelName ++ "_arity" ++ show arity ++ ".model"

  putStrLn "Testing relations:"
  results <- mapM (\((entities, p), label) -> do
                        let entityTensors = map (\e -> toDevice myDevice $ asTensor ([fromIntegral e :: Int] :: [Int])) entities
                        let rTensor = toDevice myDevice $ asTensor ([fromIntegral p :: Int] :: [Int])
                        let output = classify loadedModel Eval rTensor entityTensors
                        let confidence = asValue (fst (maxDim (Dim 1) RemoveDim output)) :: Float
                        let confThreshold = 0.5
                        let prediction = if confidence >= confThreshold then 1 else 0 :: Int
                        -- putStrLn $ "Test: " ++ show entities ++ ", " ++ show p ++ " -> Prediction: " ++ show prediction ++ " label : " ++ show label ++ " with confidence " ++ show confidence
                        -- if prediction == 1
                        --   then putStrLn "Relation holds."
                        --   else putStrLn "Relation does not hold."
                        return (label, fromIntegral prediction :: Float))
                    testRelations

  -- 精度の計算
  let correctPredictions = length $ filter (\(label, prediction) -> label == prediction) results
  let totalPredictions = length results
  let accuracy = (fromIntegral correctPredictions / fromIntegral totalPredictions) * 100 :: Double
  putStrLn $ "Accuracy: " ++ show accuracy ++ "%"
  return accuracy