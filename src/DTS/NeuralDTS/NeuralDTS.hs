{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module DTS.NeuralDTS.NeuralDTS  (
  testProcessAndTrain,
  testNeuralDTS
  ) where

import Control.Monad (forM, replicateM, forM_)
import Control.Monad.RWS (MonadState(put))
import qualified Data.Text.Lazy as T      --text
import qualified Data.Text.Lazy.IO as T   --text
import qualified Data.List as L           --base
import qualified Data.Map as Map
import qualified Data.Set as Set
import qualified Interface.Text as T
import qualified System.IO as S
import System.FilePath ((</>))
import System.Random (randomRIO, newStdGen, randomRs)
import System.Random.Shuffle (shuffle, shuffle', shuffleM)
import System.Directory (createDirectoryIfMissing)
import Debug.Trace
import System.IO.Unsafe (unsafePerformIO)
import Text.Regex.TDFA ((=~))

import DTS.NeuralDTS.PreProcess (getTrainRelations, makeNegData, writeRelationsCsv)
import DTS.NeuralDTS.Classifier.MLP (trainModel, testModel)
import qualified Parser.ChartParser as CP
import Parser.Language (jpOptions) 
import qualified Parser.Language.Japanese.Juman.CallJuman as Juman
import qualified Parser.Language.Japanese.Lexicon as L (LexicalResource(..), lexicalResourceBuilder, LexicalItems, lookupLexicon, setupLexicon, emptyCategories, myLexicon)
import qualified DTS.NeuralDTS.Classifier.Utils as Utils
import qualified DTS.NeuralDTS.Classifier.MLP as MLP (trainModel, testModel, MLPSpec(..))
import qualified DTS.NeuralDTS.Classifier.SocherNTN as SNTN (trainModel, testModel, SNTNSpec(..))
import qualified DTS.NeuralDTS.Classifier.DingNTN as DNTN (trainModel, testModel, DNTNSpec(..))
import qualified DTS.NeuralDTS.Classifier.TuckER as TuckER (trainModel, testModel, TuckERSpec(..))

inputsDir = "src/DTS/NeuralDTS/inputs"
dataDir = "src/DTS/NeuralDTS/dataSet"
imagesDir = "src/DTS/NeuralDTS/images"
modelsDir = "src/DTS/NeuralDTS/models"
indexNum = 16

checkAccuracy :: CP.ParseSetting -> [T.Text] -> IO ()
checkAccuracy ps str = do
  createDirectoryIfMissing True (dataDir </> show indexNum)
  createDirectoryIfMissing True (imagesDir </> show indexNum)
  createDirectoryIfMissing True (modelsDir </> show indexNum)

  let arities = [2]
  (posOrgRelations, posAddRelations) <- getTrainRelations ps str -- :: (Map.Map Int [([Int], Int)], Map.Map Int [([Int], Int)])
  
  ---- ファイルから読み込み
  -- posOrgRelationsList <- forM arities $ \arity -> do
  --   let filePath = dataDir </> show indexNum </> ("pos_org_relations_" ++ show arity ++ ".csv")
  --   csvLines <- readCsv filePath
  --   return $ parseRelations arity csvLines
  -- let posOrgRelations' = Map.unionsWith (++) posOrgRelationsList :: Map.Map Int [([Int], Int)]
  
  -- posAddRelationsList <- forM arities $ \arity -> do
  --   let filePath = dataDir </> show indexNum </> ("pos_add_relations_" ++ show arity ++ ".csv")
  --   csvLines <- readCsv filePath
  --   return $ parseRelations arity csvLines
  -- let posAddRelations' = Map.unionsWith (++) posAddRelationsList :: Map.Map Int [([Int], Int)]
  
  -- putStrLn "posOrgRelations Sizes:"
  -- mapM_ (\(arity, rels) -> putStrLn $ "Arity " ++ show arity ++ ": " ++ show (length rels)) (Map.toList posOrgRelations')
  -- putStrLn "posAddRelations Sizes:"
  -- mapM_ (\(arity, rels) -> putStrLn $ "Arity " ++ show arity ++ ": " ++ show (length rels)) (Map.toList posAddRelations')
  ----
  
  -- 複数回学習を行い、平均精度を出力
  forM_ arities $ \arity -> do
    ---- ファイルから読み込み
    -- entityDict <- readEntityDict (dataDir </> show indexNum </> "entity_dict_" ++ show arity ++ ".csv")
    -- putStrLn $ "entityDict: " ++ show (length entityDict)
    -- let posOrgRelations = Map.map (\relations -> groupBySentence relations entityDict) posOrgRelations' :: Map.Map Int [[([Int], Int)]]
    -- let posAddRelations = Map.map (\relations -> groupBySentence relations entityDict) posAddRelations' :: Map.Map Int [[([Int], Int)]]
    ----

    let posOrgRelationsForArity = Map.findWithDefault [] arity posOrgRelations -- :: [[([Int], Int)]]
    let posAddRelationsForArity = Map.findWithDefault [] arity posAddRelations -- :: [[([Int], Int)]]
    accuracies <- forM [1..4] $ \i -> trainAndTest i arity posOrgRelationsForArity posAddRelationsForArity
    let averageAccuracy = sum accuracies / fromIntegral (length accuracies)
    putStrLn $ "Average Accuracy for arity " ++ show arity ++ ": " ++ show averageAccuracy
  
trainAndTest :: Int -> Int -> [[([Int], Int)]] -> [[([Int], Int)]] -> IO Double
trainAndTest fold arity posOrgRelations posAddRelations = do
  -- 1. 各意味表示からtest候補を選ぶ．残りがpos training data
  let (trainPosData, testPosData) = unzip $ map splitRelations posOrgRelations
  let flatTrainPosData = concat trainPosData :: [([Int], Int)]
  let flatTestPosData = concat testPosData :: [([Int], Int)]

  -- 2. pos training dataをaugumentする
  let flatPosAddData = concat posAddRelations :: [([Int], Int)]
  let addedTrainPosData = flatTrainPosData ++ flatPosAddData :: [([Int], Int)]
  putStrLn $ "Train Data Sizes: " ++ show (length flatTrainPosData) ++ "->" ++ show (length addedTrainPosData)
  putStrLn $ "Test PosData Sizes: " ++ show (length flatTestPosData)
  
  shuffledTrainPosData <- shuffleM addedTrainPosData

  -- 3. 2.に含まれないデータをneg dataとして生成
  let allPreds = map snd addedTrainPosData
  let existingRelations = Set.fromList (shuffledTrainPosData ++ flatTestPosData)
  negData <- generateNegRelations2 shuffledTrainPosData allPreds existingRelations (length shuffledTrainPosData + length flatTestPosData)

  -- 4. 3.をneg train dataとneg test dataに分割する
  let (trainNegData, testNegData) = splitAt (length shuffledTrainPosData) negData

  shuffledTrainNegData <- shuffleM trainNegData
  shuffledTestPosData <- shuffleM flatTestPosData
  shuffledTestNegData <- shuffleM testNegData

  putStrLn "Train NegData Sizes:"
  putStrLn $ show (length shuffledTrainNegData)
  putStrLn "Test NegData Sizes:"
  putStrLn $ show (length shuffledTestNegData)

  -- 5. augumented pos training dataとneg training dataがtraining dataで、1.のpos test dataと4.のneg test dataがtest data
  let trainPosData' = map (\(xs, y) -> ((xs, y), 1.0)) shuffledTrainPosData
  let trainNegData' = map (\(xs, y) -> ((xs, y), 0.0)) shuffledTrainNegData
  let trainData = trainPosData' ++ trainNegData' :: [(([Int], Int), Float)]
  let testPosData' = map (\(xs, y) -> ((xs, y), 1.0)) shuffledTestPosData
  let testNegData' = map (\(xs, y) -> ((xs, y), 0.0)) shuffledTestNegData
  let testData = testPosData' ++ testNegData' :: [(([Int], Int), Float)]

  genTrain <- newStdGen
  genTest <- newStdGen
  let shuffledTrainData = shuffle' trainData (length trainData) genTrain
  let shuffledTestData = shuffle' testData (length testData) genTest

  writeRelationsCsv (dataDir </> show indexNum </> "train_pos_" ++ show arity ++ ".csv") shuffledTrainPosData
  writeRelationsCsv (dataDir </> show indexNum </> "train_neg_" ++ show arity ++ ".csv") shuffledTrainNegData
  writeRelationsCsv (dataDir </> show indexNum </> "test_pos_" ++ show arity ++ ".csv") shuffledTestPosData
  writeRelationsCsv (dataDir </> show indexNum </> "test_neg_" ++ show arity ++ ".csv") shuffledTestNegData

  entityCount <- Utils.getLineCount (dataDir </> show indexNum </> "entity_dict_" ++ show arity ++ ".csv")
  relationCount <- Utils.getLineCount (dataDir </> show indexNum </> "predicate_dict_" ++ show arity ++ ".csv")
  putStrLn $ "entityCount: " ++ show entityCount
  putStrLn $ "relationCount: " ++ show relationCount
  S.hFlush S.stdout
  -- MLPを使用する場合
  let mlpSpec = MLP.MLPSpec {
            entity_num_embed = entityCount,
            relation_num_embed = relationCount,
            entity_features = 256,
            relation_features = 256,
            hidden_dim1 = 216,
            hidden_dim2 = 32,
            output_feature = 1,
            arity = 2}
  let modelName = "model_arity_" ++ show arity ++ "_fold_" ++ show fold
  MLP.trainModel modelName mlpSpec shuffledTrainData arity
  accuracy <- MLP.testModel modelName mlpSpec shuffledTestData arity
  putStrLn $ "Accuracy: " ++ show accuracy
  return accuracy

-- CSVファイルを読み込む関数
readCsv :: FilePath -> IO [T.Text]
readCsv path = do
  content <- S.readFile path
  return $ T.lines (T.pack content)

-- CSVファイルに書き込む関数
writeCsv :: FilePath -> [(String, Int)] -> IO ()
writeCsv path content = do
  if null content
    then putStrLn $ "Error: No data to write to " ++ path
    else do
      let textContent = map (\(k, v) -> k ++ "," ++ show v) content
      putStrLn $ "writeCsv: " ++ path
      S.writeFile path (unlines textContent)

splitRelations :: [([Int], Int)] -> ([([Int], Int)], [([Int], Int)])
splitRelations relations = unsafePerformIO $ do
  if length relations == 1
    then return (relations, [])
    else do
      idx <- randomRIO (0, length relations - 1)
      let (before, after) = splitAt idx relations
      case after of
        [] -> do
          let (initBefore, lastElem) = splitAt (length before - 1) before
          return (initBefore, lastElem)
        (selected:rest) -> return (before ++ rest, [selected])

-- ネガティブデータを生成する関数
generateNegRelations2 :: [([Int], Int)] -> [Int] -> Set.Set ([Int], Int) -> Int -> IO [([Int], Int)]
generateNegRelations2 posRelations allPreds existingNegRelations numNegRelations = do
  let posSet = Set.fromList posRelations
  let generateOneNegRelation = do
        (entities, pred) <- randomRIO (0, length posRelations - 1) >>= \i -> return (posRelations !! i)
        newPred <- randomRIO (0, length allPreds - 1) >>= \i -> return (allPreds !! i)
        let negRelation = (entities, newPred)
        if Set.member negRelation posSet || Set.member negRelation existingNegRelations
          then generateOneNegRelation
          else return negRelation
  negRelations <- replicateM numNegRelations generateOneNegRelation
  return negRelations

parseRelations :: Int -> [T.Text] -> Map.Map Int [([Int], Int)]
parseRelations arity lines =
  Map.fromListWith (++) [(arity, [(init entities, last entities)]) | line <- lines, let entities = map (read . T.unpack) (T.splitOn (T.pack ",") line) :: [Int]]

-- エンティティ辞書の読み込み関数
readEntityDict :: FilePath -> IO (Map.Map Int String)
readEntityDict path = do
  content <- S.readFile path
  let rows = lines content
  let entityDict = Map.fromList $ map (\row -> let (entity, id) = break (== ',') row in (read (drop 1 id), entity)) rows
  return entityDict

-- 文ごとにグループ化する関数
groupBySentence :: [([Int], Int)] -> Map.Map Int String -> [[([Int], Int)]]
groupBySentence relations entityDict =
  let extractSentenceId entity =
        let entityStr = entityDict Map.! entity :: String
            -- 正規表現の型を明示的に指定
            regex :: String
            regex = "S([0-9]+)"
            match = entityStr =~ regex :: (String, String, String, [String])
        in case match of
             (_, _, _, [num]) -> read num :: Int
             _ -> 0
      sentenceMap = Map.fromListWith (++) $ map (\(entities, p) ->
          let sentenceId = maximum $ map extractSentenceId entities
          in (sentenceId, [(entities, p)])
        ) relations
  in Map.elems sentenceMap

testProcessAndTrain :: IO()
testProcessAndTrain = do
  -- CSVファイルを読み込む
  posStr <- readCsv (inputsDir ++ "/test2.csv")
  
  -- テストデータを定義
  let testStr = [T.pack "次郎が踊る"]

  lr <- L.lexicalResourceBuilder Juman.KWJA
  let ps = CP.ParseSetting jpOptions lr 1 1 1 1 True Nothing Nothing True False

  putStrLn "Start NeuralDTS"
  -- トレーニングとテストを実行
  -- processAndTrain ps posStr
  -- processAndTest ps testStr

testNeuralDTS :: IO()
testNeuralDTS = do
  -- CSVファイルを読み込む
  -- posStr <- readCsv (inputsDir ++ "/posStr.csv")
  posStr <- readCsv (inputsDir ++ "/JPWordNet.csv")
  let posStr2 = take 1000 posStr

  lr <- L.lexicalResourceBuilder Juman.KWJA
  let ps = CP.ParseSetting jpOptions lr 1 1 1 1 True Nothing Nothing True False

  -- トレーニングとテストを実行
  checkAccuracy ps posStr2