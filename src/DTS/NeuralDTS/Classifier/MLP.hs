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
  classify :: MLP -> RuntimeMode -> [Tensor] -> Tensor -> Tensor
  classify MLP {..} _ entitiesTensor predTensor =
    let entities = map (embedding' (toDependent entity_emb)) entitiesTensor
        relation = embedding' (toDependent relation_emb) predTensor
        input = cat (Dim 1) (relation : entities)
        nonlinearity = Torch.sigmoid
    in nonlinearity $ linear linear_layer3 $ nonlinearity $ linear linear_layer2 $ nonlinearity 
       $ linear linear_layer1 $ input
  -- classify :: MLP -> RuntimeMode -> [Tensor] -> Tensor -> Tensor
  -- classify MLP {..} _ entitiesTensor predTensor =
  --   let entities = map (embedding' (toDependent entity_emb)) entitiesTensor
  --       relation = embedding' (toDependent relation_emb) predTensor
  --       input = cat (Dim 1) (relation : entities)
  --       nonlinearity = Torch.sigmoid
  --       -- デバッグログを出力
  --       _ = unsafePerformIO $ do
  --         putStrLn $ "Entities shape2: " ++ show (map shape entities)
  --         putStrLn $ "Relation shape2: " ++ show (shape relation)
  --         putStrLn $ "Input shape2: " ++ show (shape input)
  --         S.hFlush S.stdout
  --   in nonlinearity $ linear linear_layer3 $ nonlinearity $ linear linear_layer2 $ nonlinearity $ linear linear_layer1 $ input

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

batchSize :: Int
batchSize = 32
numIters :: Integer
numIters = 100
myDevice :: Device
myDevice = Device CUDA 0
-- myDevice = Device CPU 0
mode :: RuntimeMode
mode = Train
lr :: LearningRate
lr = 5e-2

-- model :: MLP -> Tensor -> Tensor
-- model params t = mlp params t

saveModel :: FilePath -> MLP -> MLPSpec -> IO ()
saveModel path model spec = do
  createDirectoryIfMissing True modelsDir
  saveParams model path
  -- print "Saved Model Parameters:"
  -- print model
  B.encodeFile (path ++ "-spec") spec

loadModel :: FilePath -> MLPSpec -> IO (Maybe MLP)
loadModel path spec = do
  exists <- doesFileExist path
  if exists
    then do
      initModel <- toDevice myDevice <$> sample spec
      loadParams initModel path
      -- print "Loaded Model Parameters:"
      -- print initModel
      return $ Just initModel
    else return Nothing

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
  let prediction = squeezeAll $ classify model mode entityIdxTensors relationIdxTensor
  -- putStrLn $ "Prediction: " ++ show prediction
  return $ binaryCrossEntropyLoss' teachers prediction

trainModel :: String -> [(([Int], Int), Float)] -> Int -> IO ()
trainModel modelName trainingRelations arity = do
  putStrLn $ "trainModel : " ++ show arity
  S.hFlush S.stdout
  let (trainData, validData) = splitAt (round $ 0.8235 * fromIntegral (length trainingRelations)) trainingRelations

  entityCount <- getLineCount (dataDir </> "entity_dict" ++ show indexNum ++ ".csv")
  relationCount <- getLineCount (dataDir </> "predicate_dict" ++ show indexNum ++ ".csv")

  let checkIndexRange idx count = if idx < 0 || idx >= count then error $ "Index out of range: " ++ show idx ++ " (count: " ++ show count ++ ")" else idx
  mapM_ (\((entities, relation), _) -> do
          mapM_ (\e -> return $ checkIndexRange e entityCount) entities
          return $ checkIndexRange relation relationCount
        ) trainingRelations

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

  -- putStrLn $ "trainData shape: " ++ show (map (fst . fst) trainData)
  -- putStrLn $ "validData shape: " ++ show (map (fst . fst) validData)

  -- putStrLn $ "Entity embedding size: " ++ show (entity_num_embed spec)
  -- putStrLn $ "Relation embedding size: " ++ show (relation_num_embed spec)

  -- 学習プロセス
  let batchedTrainSet = makeBatch batchSize trainData
  let batchedValidSet = makeBatch batchSize validData
  
  -- model
  initModel <- toDevice myDevice <$> sample spec
  Torch.Train.saveParams initModel (modelsDir </> "init-model")
  print $ "model loaded"

  ((trained, _), losses) <- mapAccumM [1..numIters] (initModel, optimizer) $ \epoch (model', opt') -> do
    -- putStrLn $ "epoch #" ++ show epoch
    (batchTrained@(batchModel, _), batchLosses) <- mapAccumM batchedTrainSet (model', opt') $ 
      \batch (model, opt) -> do
        -- loss
        loss <- calculateLoss model batch
        updated <- runStep model optimizer loss lr
        -- performGC
        return (updated, asValue loss::Float)
    -- batch の長さでlossをわる
    let batchloss = sum batchLosses / (fromIntegral (length batchLosses)::Float)
    putStrLn $ "Iteration: " ++ show epoch ++ " | Loss: " ++ show batchloss
    S.hFlush S.stdout
    -- 検証データで評価
    validLosses <- mapM (calculateLoss batchModel) batchedValidSet
    let validLoss = sum validLosses / fromIntegral (length validLosses)
    putStrLn $ "Validation Loss: " ++ show (asValue validLoss :: Float)
    S.hFlush S.stdout
    return (batchTrained, (batchloss, asValue validLoss :: Float))

  -- モデルを保存
  saveModel (modelsDir </> modelName ++ "_arity" ++ show arity ++ ".model") trained spec
  putStrLn $ "Model saved to models/" ++ modelName ++ "_arity" ++ show arity ++ ".model"
  S.hFlush S.stdout

  -- 学習曲線
  let (trainLosses, validLosses) = unzip losses
  drawLearningCurve (imagesDir </> modelName ++ "_learning-curve-training_" ++ show arity ++ ".png") "Learning Curve" [("", reverse trainLosses)]
  drawLearningCurve (imagesDir </> modelName ++ "_learning-curve-valid_" ++ show arity ++ ".png") "Learning Curve" [("", reverse validLosses)]
  putStrLn $ "drawLearningCurve"

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

  model <- loadModel (modelsDir </> modelName ++ "_arity" ++ show arity ++ ".model") spec
  putStrLn $ "Model loaded from models/" ++ modelName ++ "_arity" ++ show arity ++ ".model"

  case model of
    Nothing -> putStrLn $ "Model not found: models/" ++ modelName ++ "_arity" ++ show arity ++ ".model"
    Just trained -> do
      -- テストデータの判定
      putStrLn "Testing relations:"
      results <- mapM (\((entities, p), label) -> do
                         let entityTensors = map (\e -> toDevice myDevice $ asTensor ([fromIntegral e :: Int] :: [Int])) entities
                         let rTensor = toDevice myDevice $ asTensor ([fromIntegral p :: Int] :: [Int])
                         let output = classify trained Eval entityTensors rTensor
                         let confidence = asValue (fst (maxDim (Dim 1) RemoveDim output)) :: Float
                         let confThreshold = 0.5
                         let prediction = if confidence >= confThreshold then 1 else 0
                         putStrLn $ "Test: " ++ show entities ++ ", " ++ show p ++ " -> Prediction: " ++ show prediction ++ " label : " ++ show label ++ " with confidence " ++ show confidence
                         putStrLn $ "Output tensor: " ++ show output
                         putStrLn $ "Confidence: " ++ show confidence
                         putStrLn $ "prediction: " ++ show prediction
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