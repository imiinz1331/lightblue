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
import Data.List (transpose)
import Debug.Trace as D
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

modelsDir = "src/DTS/NeuralDTS/models"
dataDir = "src/DTS/NeuralDTS/dataSet"
imagesDir = "src/DTS/NeuralDTS/images"
indexNum = 13

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
    tensor_layer1 :: Parameter,
    tensor_layer2 :: Parameter,
    tensor_layer3 :: Parameter,
    linear_layer1 :: LinearParams,
    linear_layer2 :: LinearParams,
    linear_layer3 :: LinearParams,
    linear_layer :: LinearParams,
    dropout_prob :: Double
  } deriving (Generic, Parameterized)

instance Randomizable NTNSpec NTN where
  sample NTNSpec {..} = NTN
    <$> (makeIndependent =<< randnIO' [entity_num_embed, embedding_features])
    <*> (makeIndependent =<< randnIO' [relation_num_embed, embedding_features])
    <*> (makeIndependent =<< randnIO' [embedding_features, embedding_features, tensor_dim])
    <*> (makeIndependent =<< randnIO' [embedding_features, embedding_features, tensor_dim])
    <*> (makeIndependent =<< randnIO' [tensor_dim, tensor_dim, tensor_dim])
    <*> sample (LinearHypParams myDevice True (embedding_features * 2) tensor_dim)
    <*> sample (LinearHypParams myDevice True (embedding_features * 2) tensor_dim)
    <*> sample (LinearHypParams myDevice True (tensor_dim * 2) tensor_dim)
    <*> sample (LinearHypParams myDevice True tensor_dim 1)
    <*> return dropout_probability

instance Classifier NTN where
  classify :: NTN -> RuntimeMode -> Tensor -> [Tensor] -> Tensor
  classify NTN {..} mode relation_idxes [entity1_idxes, entity2_idxes] =
    let entity1 = embedding' (toDependent entity_emb) entity1_idxes
        relation = embedding' (toDependent relation_emb) relation_idxes
        entity2 = embedding' (toDependent entity_emb) entity2_idxes
        r1 = Torch.tanh $ tensorLinear entity1 relation linear_layer1 tensor_layer1
        r2 = Torch.tanh $ tensorLinear relation entity2 linear_layer2 tensor_layer2
        u = Torch.tanh $ tensorLinear r1 r2 linear_layer3 tensor_layer3
        -- output = Torch.sigmoid $ linearLayer linear_layer u
        u_dropped = if mode == Train then unsafePerformIO $ dropout dropout_prob True u else u
        output = Torch.sigmoid $ linearLayer linear_layer u_dropped
    in output
    
tensorLinear :: Tensor -> Tensor -> LinearParams -> Parameter -> Tensor
tensorLinear o1 o2 linear_layer tensor_layer = 
  let batchSize2 = shape o1 !! 0
      d = shape o1 !! 1
      o1_o2 = cat (Dim 1) [o1, o2]
      linear_product = linearLayer linear_layer o1_o2
      o1_tensor = mm o1 (view [d, -1] (toDependent tensor_layer))
      -- |tensor_product| = (batch_size, tensor_dim, 1)
      tensor_product = bmm (view [batchSize2, -1, d] o1_tensor) (contiguous $ permute [0, 2, 1] $ unsqueeze (Dim 1) o2)
      -- |tensor_product_shaped| = (batch_size, tensor_dim)
      tensor_product_shaped = reshape [shape tensor_product !! 0, shape tensor_product !! 1] tensor_product
  in tensor_product_shaped + linear_product

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

batchSize :: Int
batchSize = 256
numIters :: Integer
numIters = 500
myDevice :: Device
myDevice = Device CUDA 0
mode :: RuntimeMode
mode = Train
lr :: LearningRate
lr = 5e-3

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
  let optimizer = GD
--   let optimizer = mkAdam 0 0.9 0.999 (flattenParameters initModel)
  
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