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

import DTS.NeuralDTS.Classifier

modelsDir = "src/DTS/NeuralDTS/models"
dataDir = "src/DTS/NeuralDTS/dataSet"
imagesDir = "src/DTS/NeuralDTS/images"

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

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

batchSize :: Int
batchSize = 32
numIters :: Integer
numIters = 100
myDevice :: Device
myDevice = Device CUDA 0
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
  B.encodeFile (path ++ "-spec") spec

loadModel :: FilePath -> MLPSpec -> IO (Maybe MLP)
loadModel path spec = do
  exists <- doesFileExist path
  if exists
    then do
      initModel <- toDevice myDevice <$> sample spec
      loadParams initModel path
      print "model loaded"
      return $ Just initModel
    else return Nothing

getLineCount :: FilePath -> IO Int
getLineCount path = do
  content <- readFile path
  return $ length (lines content)

calculateLoss :: (Classifier n) => n -> [(([Int], Int), Float)] -> IO Tensor
calculateLoss model dataSet = do
  let (input, label) = unzip dataSet
      -- inputs
      (entityIdxes, relationIdxes) = unzip input :: ([[Int]], [Int])
      -- 正解データ
      teachers = toDevice myDevice $ asTensor (label :: [Float]) :: Tensor
      -- 予測データ
      entityIdxTensors = map (toDevice myDevice . asTensor) (Data.List.transpose entityIdxes)
      relationIdxTensor = toDevice myDevice $ asTensor (relationIdxes :: [Int])
      prediction = squeezeAll $ classify model mode entityIdxTensors relationIdxTensor
  return $ binaryCrossEntropyLoss' teachers prediction

data2Idx :: [(([Int], Int), Float)] -> ([[Int]], [Int])
data2Idx = unzip . map fst

splitData :: [a] -> Double -> IO ([a], [a])
splitData xs ratio = do
  let n = round $ ratio * fromIntegral (length xs)
  indices <- replicateM (length xs) (randomRIO (0, length xs - 1))
  let (trainIndices, validIndices) = splitAt n indices
  return (map (xs !!) trainIndices, map (xs !!) validIndices)

trainModel :: String -> [(([Int], Int), Float)] -> Int -> IO ()
trainModel modelName trainingRelations arity = do
  putStrLn $ "trainModel : " ++ show arity
  (trainData, validData) <- splitData trainingRelations 0.8

  entityCount <- getLineCount (dataDir </> "entity_dict.csv")
  relationCount <- getLineCount (dataDir </> "predicate_dict.csv")

  -- モデルの設定
  let spec = MLPSpec
        { entity_num_embed = entityCount  -- entityの埋め込み数
        , relation_num_embed = relationCount  -- 関係の埋め込み数
        , entity_features = 64  -- entityの特徴量数
        , relation_features = 32  -- 関係の特徴量数
        , hidden_dim1 = 128  -- 最初の隠れ層の次元数
        , hidden_dim2 = 64  -- 2番目の隠れ層の次元数
        , output_feature = 2  -- 出力の特徴量数
        , arity = arity
        }

  initModel <- toDevice myDevice <$> sample spec

  -- データが空でないことを確認
  when (null trainData && null validData) $ do
    error "Training data or labels are empty. Check your input data."

  putStrLn $ "trainData shape: " ++ show (map (fst . fst) trainData)
  putStrLn $ "validData shape: " ++ show (map (fst . fst) validData)

  -- 学習プロセス
  let batchedTrainSet = makeBatch batchSize trainData
  let batchedValidSet = makeBatch batchSize validData
  ((trained, _), losses) <- mapAccumM [1..numIters] (initModel, optimizer) $ \epoch (model', opt') -> do
    -- putStrLn $ "epoch #" ++ show epoch
    (batchTrained@(batchModel, _), batchLosses) <- mapAccumM batchedTrainSet (model', opt') $ 
      \batch (model, opt) -> do
        -- loss
        loss <- calculateLoss model batch
        updated <- runStep model optimizer loss lr
        return (updated, asValue loss::Float)
    -- batch の長さでlossをわる
    let batchloss = sum batchLosses / (fromIntegral (length batchLosses)::Float)
    putStrLn $ "Iteration: " ++ show epoch ++ " | Loss: " ++ show batchloss
    -- 検証データで評価
    validLosses <- mapM (calculateLoss batchModel) batchedValidSet
    let validLoss = sum validLosses / fromIntegral (length validLosses)
    putStrLn $ "Validation Loss: " ++ show (asValue validLoss :: Float)
    return (batchTrained, (batchloss, asValue validLoss :: Float))

  -- モデルを保存
  saveModel (modelsDir </> modelName ++ "_arity" ++ show arity ++ ".model") trained spec
  putStrLn $ "Model saved to models/" ++ modelName ++ "_arity" ++ show arity ++ ".model"

  -- TODO : 学習曲線
  let (trainLosses, validLosses) = unzip losses
  drawLearningCurve (imagesDir </> modelName ++ "learning-curve-training.png") "Learning Curve" [("", reverse trainLosses)]
  drawLearningCurve (imagesDir </> modelName ++ "learning-curve-valid.png") "Learning Curve" [("", reverse validLosses)]

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

testModel :: String -> [([Int], Int)] -> Int -> IO ()
testModel modelName testRelations arity = do
  putStrLn "testModel"

  entityCount <- getLineCount (dataDir </> "entity_dict.csv")
  relationCount <- getLineCount (dataDir </> "predicate_dict.csv")
  let spec = MLPSpec
        { entity_num_embed = entityCount  -- entityの埋め込み数
        , relation_num_embed = relationCount  -- 関係の埋め込み数
        , entity_features = 64  -- entityの特徴量数
        , relation_features = 32  -- 関係の特徴量数
        , hidden_dim1 = 128  -- 最初の隠れ層の次元数
        , hidden_dim2 = 64  -- 2番目の隠れ層の次元数
        , output_feature = 2  -- 出力の特徴量数
        , arity = arity
        }

  model <- loadModel (modelsDir </> modelName ++ "_arity" ++ show arity ++ ".model") spec
  putStrLn $ "Model loaded from models/" ++ modelName ++ "_arity" ++ show arity ++ ".model"

  case model of
    Nothing -> putStrLn $ "Model not found: models/" ++ modelName ++ ".model"
    Just trained -> do
      -- テストデータの判定
      putStrLn "Testing relations:"
      mapM_ (\(entities, p) -> do
               let entityTensors = map (\e -> toDevice myDevice $ asTensor ([fromIntegral e :: Int] :: [Int])) entities
               let rTensor = toDevice myDevice $ asTensor ([fromIntegral p :: Int] :: [Int])
               let output = classify trained Eval entityTensors rTensor
               let prediction = asValue (argmax (Dim 1) RemoveDim output) :: Int
               let confidence = asValue (fst (maxDim (Dim 1) RemoveDim output)) :: Float
               putStrLn $ "Test: " ++ show entities ++ " -> Prediction: " ++ show prediction ++ " with confidence " ++ show confidence
               if prediction == 1
                 then putStrLn "Relation holds."
                 else putStrLn "Relation does not hold.")
            testRelations