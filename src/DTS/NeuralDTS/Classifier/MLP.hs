{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
-- {-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module DTS.NeuralDTS.Classifier.MLP (
  trainModel,
  testModel
  ) where

import Control.Monad (when)
import Control.Monad.Cont (replicateM)
import Data.List (foldl', intersperse, scanl')
import qualified Data.ByteString as B
import qualified Data.Binary as B
import GHC.Generics ( Generic )
import Torch
import Torch.Control (mapAccumM, makeBatch)
import qualified Torch.Train
import System.Directory (createDirectoryIfMissing, doesFileExist)
import System.Random (randomRIO)
import System.FilePath ((</>))

import DTS.NeuralDTS.Classifier

modelsDir = "src/DTS/NeuralDTS/models"
dataDir = "src/DTS/NeuralDTS/DataSet"

data MLPSpec = MLPSpec
  { 
    entity_num_embed :: Int,
    relation_num_embed :: Int,
    entity_features :: Int,
    relation_features :: Int,
    hidden_dim1 :: Int,
    hidden_dim2 :: Int,
    output_feature :: Int
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
    <*> sample (LinearSpec (entity_features * 2 + relation_features) hidden_dim1) -- 最初の線形層の入力サイズ 160
    <*> sample (LinearSpec hidden_dim1 hidden_dim2)
    <*> sample (LinearSpec hidden_dim2 output_feature)

instance Classifier MLP where
  classify MLP {..} _ entity1_idxes relation_idxes entity2_idxes =
    let _ = trace "classify function called" "" `seq` ()
        entity1 = embedding' (toDependent entity_emb) entity1_idxes
        _ = trace ("entity1 size: " ++ show (shape entity1)) "" `seq` ()
        relation = embedding' (toDependent relation_emb) relation_idxes
        _ = trace ("relation size: " ++ show (shape relation)) "" `seq` ()
        entity2 = embedding' (toDependent entity_emb) entity2_idxes
        _ = trace ("entity2 size: " ++ show (shape entity2)) "" `seq` ()
        input = cat (Dim 1) [entity1, relation, entity2]
        _ = trace ("input size: " ++ show (shape input)) "" `seq` ()
        nonlinearity = Torch.sigmoid
    in nonlinearity $ linear linear_layer3 $ nonlinearity $ linear linear_layer2 $ nonlinearity 
       $ linear linear_layer1 $ input

mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = foldl' revApply input $ intersperse relu [linear linear_layer1, linear linear_layer2, linear linear_layer3]
  where
    revApply x f = f x

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

batchSize = 2
numIters = 100
myDevice = Device CUDA 0
mode = Train
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

calculateLoss :: (Classifier n) => n -> [(((Int, Int), Int), Float)] -> Tensor
calculateLoss model dataSet =
  let (input, label) = unzip dataSet
      -- inputs
      (entity1Idx, relationIdx, entity2Idx) = data2Idx input
      -- 正解データ
      teachers = toDevice myDevice $ asTensor (label :: [Float])
      -- 予測データ
      entity1IdxTensor = toDevice myDevice $ asTensor (entity1Idx :: [Int])
      relationIdxTensor = toDevice myDevice $ asTensor (relationIdx :: [Int])
      entity2IdxTensor = toDevice myDevice $ asTensor (entity2Idx :: [Int])
      prediction = squeezeAll $ classify model mode entity1IdxTensor relationIdxTensor entity2IdxTensor
      -- _ = trace ("Teachers: " ++ show teachers) teachers `seq` ()
      -- _ = trace ("Prediction: " ++ show prediction) prediction `seq` ()
  in binaryCrossEntropyLoss' teachers prediction

data2Idx :: [((Int, Int), Int)] -> ([Int], [Int], [Int])
data2Idx = unzip3 . map idx
  where
    idx :: ((Int, Int), Int) -> (Int, Int, Int)
    idx ((e1, e2), r) = (e1, r, e2)

trainModel :: String -> [(((Int, Int), Int), Float)] -> IO ()
trainModel modelName trainingRelations = do
  putStrLn $ "trainModel"

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
        }

  initModel <- toDevice myDevice <$> sample spec

  -- データが空でないことを確認
  when (null trainingRelations) $ do
    error "Training data or labels are empty. Check your input data."

  -- 学習プロセス
  let batchedTrainSet = makeBatch batchSize trainingRelations
  ((trained, _), losses) <- mapAccumM [1..numIters] (initModel, optimizer) $ \epoch (model', opt') -> do
    -- putStrLn $ "epoch #" ++ show epoch
    (batchTrained@(batchModel, _), batchLosses) <- mapAccumM batchedTrainSet (model', opt') $ 
      \batch (model, opt) -> do
        -- loss
        let loss = calculateLoss model batch
        updated <- runStep model optimizer loss lr
        return (updated, asValue loss::Float)
    -- batch の長さでlossをわる
    let batchloss = sum batchLosses / (fromIntegral (length batchLosses)::Float)
    putStrLn $ "Iteration: " ++ show epoch ++ " | Loss: " ++ show batchloss
    -- TODO : validation Data
    -- let validLoss = calculateLoss batchModel validSet
    let validLoss = 0.0
    return (batchTrained, (batchloss, asValue validLoss::Float))

  -- モデルを保存
  saveModel (modelsDir </> modelName ++ ".model") trained spec
  putStrLn $ "Model saved to models/" ++ modelName ++ ".model"

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

testModel :: String -> [((Int, Int), Int)] -> IO ()
testModel modelName testRelations = do
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
        }

  maybeModel <- loadModel (modelsDir </> modelName ++ ".model") spec
  case maybeModel of
    Nothing -> putStrLn $ "Model not found: models/" ++ modelName ++ ".model"
    Just trained -> do
      -- テストデータの判定
      putStrLn "Testing relations:"
      mapM_ (\((e1, e2), p) -> do
               let e1Tensor = toDevice myDevice $ asTensor ([fromIntegral e1 :: Int] :: [Int])
               let e2Tensor = toDevice myDevice $ asTensor ([fromIntegral e2 :: Int] :: [Int])
               let rTensor = toDevice myDevice $ asTensor ([fromIntegral p :: Int] :: [Int])
               let output = classify trained Eval e1Tensor rTensor e2Tensor
               let prediction = asValue (argmax (Dim 1) RemoveDim output) :: Int
               let (confidenceTensor, _) = maxDim (Dim 1) RemoveDim output
               let confidence = asValue confidenceTensor :: Float
               putStrLn $ "Test: (" ++ show e1 ++ ", " ++ show e2 ++ ", " ++ show p ++ ") -> Prediction: " ++ show prediction ++ " with confidence " ++ show confidence
               if prediction == 1
                 then putStrLn "Relation holds."
                 else putStrLn "Relation does not hold.")
            testRelations