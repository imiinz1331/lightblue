{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE InstanceSigs #-}

{- Author: Daisuke Bekki -}

module DTS.NeuralDTS.Classifier.SocherNTN where

import Control.Monad (when)
import Data.Binary as B
import Data.Function    ((&))
import Data.List (transpose)
import GHC.Generics
import System.FilePath ((</>))
import Prelude hiding   (tanh) 
import qualified System.IO as S
import System.IO.Unsafe (unsafePerformIO)
import System.Directory (createDirectoryIfMissing)

import Torch
import Debug.Trace as D
import Torch.Control (mapAccumM, makeBatch)
import Torch.Tensor      (Tensor(..),shape,select,sliceDim,reshape)
import Torch.Functional  (Dim(..),embedding',matmul,transpose,tanh,cat,squeezeAll)
import Torch.Device      (Device(..))
import Torch.NN          (Parameterized(..),Randomizable(..),Parameter,sample)
import Torch.Autograd    (IndependentTensor(..),makeIndependent)
import Torch.Tensor.Initializers    (xavierUniform')
import qualified Torch.Train
import ML.Exp.Chart (drawLearningCurve)

import DTS.NeuralDTS.Classifier (Classifier(..))
import qualified Data.Type.Bool as D

modelsDir = "src/DTS/NeuralDTS/models"
dataDir = "src/DTS/NeuralDTS/dataSet"
imagesDir = "src/DTS/NeuralDTS/images"
indexNum = 11

data SNTNSpec = SNTNSpec {
  entity_num_embed :: Int
  , relation_num_embed :: Int
  , embedding_features :: Int
  , output_dim :: Int
  } deriving (Generic, Binary, Show)

data SNTN = SNTN {
  entityEmb :: Parameter
  , relWEmb :: Parameter
  , relVEmb :: Parameter
  , relBEmb :: Parameter
  , relUEmb :: Parameter
  } deriving (Generic, Parameterized)

instance Randomizable SNTNSpec SNTN where
  sample SNTNSpec{..} = do
    let relationWDim = embedding_features * embedding_features * output_dim
        relationVDim = 2 * embedding_features * output_dim
    SNTN
      <$> (makeIndependent =<< xavierUniform' myDevice [entity_num_embed, embedding_features]) -- entityEmb
      <*> (makeIndependent =<< xavierUniform' myDevice [relation_num_embed, relationWDim])    -- relWEmb
      <*> (makeIndependent =<< xavierUniform' myDevice [relation_num_embed, relationVDim])    -- relVEmb
      <*> (makeIndependent =<< xavierUniform' myDevice [relation_num_embed, output_dim])      -- relBEmb
      <*> (makeIndependent =<< xavierUniform' myDevice [relation_num_embed, output_dim])      -- relUEmb

-- | Backwards function application.
(.->) :: a -> (a -> b) -> b
(.->) = (&)

instance Classifier SNTN where
  classify :: SNTN -> RuntimeMode -> Tensor -> [Tensor] -> Tensor
  classify SNTN{..} _ relation_idxes [entity1_idxes, entity2_idxes] =
    let [nBatch] = shape entity1_idxes
        [_,embedDim'] = shape $ toDependent entityEmb
        [_,nHeads'] = shape $ toDependent $ relBEmb
        -- エンティティ埋め込み
        e1 = embedding' (toDependent entityEmb) entity1_idxes       -- | <nBatch,embedDim>
        e2 = embedding' (toDependent entityEmb) entity2_idxes       -- | <nBatch,embedDim>
       -- 関係埋め込み
        relWBatch = embedding' (toDependent relWEmb) relation_idxes -- Shape: <nBatch, embedDim*embedDim*nHeads>
        relW = let expectedSize = nBatch * nHeads' * embedDim' * embedDim'
                   actualSize = product $ shape relWBatch
               in if actualSize /= expectedSize
                     then error $ "relW reshape size mismatch: expected " ++ show expectedSize ++ ", got " ++ show actualSize
                     else reshape [nBatch, nHeads', embedDim', embedDim'] relWBatch -- Shape: <nBatch, nHeads, embedDim, embedDim>
        relVBatch = embedding' (toDependent relVEmb) relation_idxes -- Shape: <nBatch, 2*embedDim*nHeads>
        relV = reshape [nBatch, nHeads', 2 * embedDim'] relVBatch -- Shape: <nBatch, nHeads, 2*embedDim>
        relBBatch = embedding' (toDependent relBEmb) relation_idxes -- Shape: <nBatch, nHeads>
        relB = reshape [nBatch, nHeads', 1] relBBatch -- Shape: <nBatch, nHeads, 1>
        relUBatch = embedding' (toDependent relUEmb) relation_idxes -- Shape: <nBatch, nHeads>
        relU = reshape [nBatch, 1, nHeads'] relUBatch -- Shape: <nBatch, 1, nHeads>
        -- nPartの計算
        -- e1Reshaped = reshape [nBatch, 1, embedDim', 1] e1           -- Shape: <nBatch, 1, embedDim, 1>
        -- relWReshaped = reshape [nBatch, nHeads', embedDim', embedDim'] relW -- Shape: <nBatch, nHeads, embedDim, embedDim>
        -- e2Reshaped = reshape [nBatch, 1, 1, embedDim'] e2           -- Shape: <nBatch, 1, 1, embedDim>
        -- nPartIntermediate = matmul e1Reshaped relW         -- Shape: <nBatch, nHeads, embedDim, 1>
        -- nPartFinal = matmul nPartIntermediate e2Reshaped           -- Shape: <nBatch, nHeads, 1, 1>
        -- nPart = reshape [nBatch, nHeads', 1] nPartFinal            -- Shape: <nBatch, nHeads, 1>
        -- nPartの計算
        nPart = e1                                    -- | <nBatch,embedDim>
             .-> reshape [nBatch,1,embedDim',1]       -- | <nBatch,1,embedDim,1>
             .-> matmul relW                          -- | <nBatch,nHeads,embedDim,1>
             .-> matmul (reshape [nBatch,1,1,embedDim'] e2) -- | <nBatch,nHeads,1,1>
             .-> reshape [nBatch,nHeads',1]           -- | <nBatch,nHeads,1>
        -- vPartの計算
        e1e2Concat = cat (Dim 1) [e1, e2] -- Shape: <nBatch, 2*embedDim>
        e1e2Reshaped = reshape [nBatch, 2 * embedDim', 1] e1e2Concat -- Shape: <nBatch, 2*embedDim, 1>
        vPart = matmul relV e1e2Reshaped -- Shape: <nBatch, nHeads, 1>
        -- NTNのスコア計算
        summedParts = nPart + vPart + relB -- Shape: <nBatch, nHeads, 1>
        activated = tanh summedParts -- Shape: <nBatch, nHeads, 1>
        output = matmul relU activated -- Shape: <nBatch, 1, nHeads>
        -- outputReshaped = reshape [nBatch, nHeads'] output -- Shape: <nBatch, nHeads>
    in sigmoid output

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

batchSize :: Int
batchSize = 128
numIters :: Integer
numIters = 1000
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

trainModel :: String -> SNTNSpec -> [(([Int], Int), Float)] -> Int -> IO ()
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

testModel :: String -> SNTNSpec -> [(([Int], Int), Float)] -> Int -> IO Double
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