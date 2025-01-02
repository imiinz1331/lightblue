{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
-- {-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE InstanceSigs #-}

module DTS.NeuralDTS.Classifier.MLP (
  trainModel,
  testModel
  ) where

import Control.Monad (when)
import Control.Monad.Cont (replicateM)
import Data.List (foldl', intersperse, scanl', transpose)
import qualified Data.ByteString as B
import qualified Data.Binary as B
import GHC.Generics ( Generic )
import ML.Exp.Chart (drawLearningCurve)
import Torch
import Torch.Control (mapAccumM, makeBatch)
import qualified Torch.Train
import System.Directory (createDirectoryIfMissing, doesFileExist)
import System.Random (randomRIO)
import System.FilePath ((</>))
import System.Mem (performGC)
import qualified System.IO as S
import System.IO.Unsafe (unsafePerformIO)
import System.IO (withFile, IOMode(..), hFileSize)

import DTS.NeuralDTS.Classifier

modelsDir = "src/DTS/NeuralDTS/models"
dataDir = "src/DTS/NeuralDTS/dataSet"
imagesDir = "src/DTS/NeuralDTS/images"
indexNum = 7

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
  } deriving (Generic, B.Binary)

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
numIters = 500
myDevice :: Device
myDevice = Device CUDA 0
-- myDevice = Device CPU 0
mode :: RuntimeMode
mode = Train
lr :: LearningRate
lr = 5e-2

getLineCount :: FilePath -> IO Int
getLineCount path = do
  content <- readFile path
  return $ length (lines content)

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

trainModel :: String -> [(([Int], Int), Float)] -> [(([Int], Int), Float)] -> Int -> IO ()
trainModel modelName trainData validData arity = do
  putStrLn $ "trainModel : " ++ show arity
  S.hFlush S.stdout

  entityCount <- getLineCount (dataDir </> "entity_dict" ++ show indexNum ++ ".csv")
  relationCount <- getLineCount (dataDir </> "predicate_dict" ++ show indexNum ++ ".csv")
  print $ "entityCount: " ++ show entityCount
  print $ "relationCount: " ++ show relationCount
  S.hFlush S.stdout

  -- モデルの設定
  let spec = MLPSpec
        { entity_num_embed = entityCount  -- entityの埋め込み数
        , relation_num_embed = relationCount  -- 関係の埋め込み数
        , entity_features = 128  -- entityの特徴量数
        , relation_features = 64  -- 関係の特徴量数
        , hidden_dim1 = 128  -- 最初の隠れ層の次元数
        , hidden_dim2 = 32  -- 2番目の隠れ層の次元数
        , output_feature = 1  -- 出力の特徴量数
        , arity = arity
        }


  -- データが空でないことを確認
  when (null trainData && null validData) $ do
    error "Training data or labels are empty. Check your input data."

  -- 学習プロセス
  let batchedTrainSet = makeBatch batchSize trainData
  let batchedValidSet = makeBatch batchSize validData
  
  -- model
  initModel <- toDevice myDevice <$> sample spec
  
  ((trainedModel, _), losses) <- mapAccumM [1..numIters] (initModel, optimizer) $ \epoch (model', opt') -> do
    -- putStrLn $ "epoch #" ++ show epoch
    (batchTrained@(batchModel, _), batchLosses) <- mapAccumM batchedTrainSet (model', opt') $ 
      \batch (model, opt) -> do
        -- loss
        loss <- calculateLoss model batch
        updated <- runStep model opt loss lr
        -- performGC
        return (updated, asValue loss::Float)
    -- batch の長さでlossをわる
    let batchloss = sum batchLosses / (fromIntegral (length batchLosses)::Float)
    -- 検証データで評価
    validLosses <- mapM (calculateLoss batchModel) batchedValidSet
    let validLoss = sum validLosses / fromIntegral (length validLosses)
    when (epoch `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show epoch ++ " | Loss: " ++ show batchloss
      putStrLn $ "Validation Loss: " ++ show (asValue validLoss :: Float)
      S.hFlush S.stdout
    return (batchTrained, (batchloss, asValue validLoss :: Float))

  -- モデルを保存
  Torch.Train.saveParams trainedModel (modelsDir </> modelName ++ "_arity" ++ show arity ++ ".model")
  B.encodeFile (modelsDir </> modelName ++ "_arity" ++ show arity ++ ".model-spec") spec
  putStrLn $ "Model saved to models/" ++ modelName ++ "_arity" ++ show arity ++ ".model"
  S.hFlush S.stdout

  -- 学習曲線
  let (trainLosses, validLosses) = unzip losses
  drawLearningCurve (imagesDir </> modelName ++ "_learning-curve-training_" ++ show arity ++ ".png") "Learning Curve" [("", reverse trainLosses)]
  drawLearningCurve (imagesDir </> modelName ++ "_learning-curve-valid_" ++ show arity ++ ".png") "Learning Curve" [("", reverse validLosses)]
  putStrLn "drawLearningCurve"

  where
    optimizer = GD

    randSelect :: [a] -> Int -> IO [a]
    randSelect xs n = do
      indices <- replicateM n $ randomRIO (0, length xs - 1)
      return [xs !! i | i <- indices]

    makeBatch :: Int -> [a] -> [[a]]
    makeBatch size xs = case splitAt size xs of
      ([], _) -> []
      (batch, rest) -> batch : makeBatch size rest

testModel :: String -> [(([Int], Int), Float)] -> Int -> IO ()
testModel modelName testRelations arity = do
  putStrLn "testModel"

  entityCount <- getLineCount (dataDir </> "entity_dict" ++ show indexNum ++ ".csv")
  relationCount <- getLineCount (dataDir </> "predicate_dict" ++ show indexNum ++ ".csv")
  let spec = MLPSpec
        { entity_num_embed = entityCount  -- entityの埋め込み数
        , relation_num_embed = relationCount  -- 関係の埋め込み数
        , entity_features = 128  -- entityの特徴量数
        , relation_features = 64  -- 関係の特徴量数
        , hidden_dim1 = 128  -- 最初の隠れ層の次元数
        , hidden_dim2 = 32  -- 2番目の隠れ層の次元数
        , output_feature = 1  -- 出力の特徴量数
        , arity = arity
        }

  loadedModel <- Torch.Train.loadParams spec (modelsDir </> modelName ++ "_arity" ++ show arity ++ ".model")
  putStrLn $ "Model loaded from models/" ++ modelName ++ "_arity" ++ show arity ++ ".model"

  putStrLn "Testing relations:"
  results <- mapM (\((entities, p), label) -> do
                        let entityTensors = map (\e -> toDevice myDevice $ asTensor ([fromIntegral e :: Int] :: [Int])) entities
                        let rTensor = toDevice myDevice $ asTensor ([fromIntegral p :: Int] :: [Int])
                        let output = classify loadedModel Eval rTensor entityTensors
                        let confidence = asValue (fst (maxDim (Dim 1) RemoveDim output)) :: Float
                        let confThreshold = 0.5
                        let prediction = if confidence >= confThreshold then 1 else 0
                        putStrLn $ "Test: " ++ show entities ++ ", " ++ show p ++ " -> Prediction: " ++ show prediction ++ " label : " ++ show label ++ " with confidence " ++ show confidence
                        if prediction == 1
                          then putStrLn "Relation holds."
                          else putStrLn "Relation does not hold."
                        return (label, fromIntegral prediction :: Float))
                    testRelations

  -- 精度の計算
  let correctPredictions = length $ filter (\(label, prediction) -> label == prediction) results
  let totalPredictions = length results
  let accuracy = (fromIntegral correctPredictions / fromIntegral totalPredictions) * 100 :: Double
  putStrLn $ "Accuracy: " ++ show accuracy ++ "%"