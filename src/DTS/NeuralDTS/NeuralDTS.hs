{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module DTS.NeuralDTS.NeuralDTS  (
  testNeuralDTS
  ) where

import Control.Monad (forM, replicateM, forM_)
import qualified Data.Text.Lazy as T      --text
import qualified Data.Text.Lazy.IO as T   --text
import qualified Data.List as L           --base
import qualified Data.Map as Map
import Data.Maybe (mapMaybe)
import qualified Data.Set as Set
import qualified Interface.Text as T
import qualified System.IO as S
import System.FilePath ((</>))
import System.Random (randomRIO, newStdGen)
import System.Random.Shuffle (shuffle', shuffleM)
import System.Directory (createDirectoryIfMissing)
import Debug.Trace
import System.IO.Unsafe (unsafePerformIO)
import Text.Regex.TDFA ((=~))

import DTS.NeuralDTS.PreProcess (getTrainRelations, writeRelationsCsv)
import qualified Parser.ChartParser as CP
import Parser.Language (jpOptions) 
import qualified Parser.Language.Japanese.Juman.CallJuman as Juman
import qualified Parser.Language.Japanese.Lexicon as L (lexicalResourceBuilder)
import qualified DTS.NeuralDTS.Classifier.Utils as Utils
import qualified DTS.NeuralDTS.Classifier.MLP as MLP (trainModel, testModel, MLPSpec(..))
import qualified DTS.NeuralDTS.Classifier.SocherNTN as SNTN (trainModel, testModel, SNTNSpec(..))
-- import qualified DTS.NeuralDTS.Classifier.DingNTN as DNTN (trainModel, testModel, DNTNSpec(..))
import qualified DTS.NeuralDTS.Classifier.TuckER as TuckER (trainModel, testModel, TuckERSpec(..))
import qualified DTS.NeuralDTS.Classifier.MLP2 as MLP2 (trainModel, testModel, MLPSpec(..))
import Control.Monad.RWS (MonadState(put))

inputsDir = "src/DTS/NeuralDTS/inputs"
dataDir = "src/DTS/NeuralDTS/dataSet"
imagesDir = "src/DTS/NeuralDTS/images"
modelsDir = "src/DTS/NeuralDTS/models"
indexNum = 37

testNeuralDTS :: IO()
testNeuralDTS = do
  -- CSVファイルを読み込む
  -- posStr <- readCsv (inputsDir ++ "/posStr.csv")
  -- posStr <- readCsv (inputsDir ++ "/JPWordNet.csv")
  posStr <- readCsv (inputsDir ++ "/yasashii_japanese.csv")
  -- posStr <- readCsv (inputsDir ++ "/john_yasashii_japanese.csv")
  let posStr2 = take 10 posStr
  -- let posStr2 = take 100 (drop 900 posStr)

  lr <- L.lexicalResourceBuilder Juman.KWJA
  let ps = CP.ParseSetting jpOptions lr 1 1 1 1 True Nothing Nothing True False

  -- トレーニングとテストを実行
  checkAccuracy ps posStr2

checkAccuracy :: CP.ParseSetting -> [T.Text] -> IO ()
checkAccuracy ps str = do
  createDirectoryIfMissing True (dataDir </> show indexNum)
  createDirectoryIfMissing True (imagesDir </> show indexNum)
  createDirectoryIfMissing True (modelsDir </> show indexNum)

  let arities = [2]
  -- (posOrgRelations, posAddRelations) <- getTrainRelations ps str -- :: (Map.Map Int [([Int], Int)], Map.Map Int [([Int], Int)])
  
  ---- ファイルから読み込み
  posOrgRelationsList <- forM arities $ \arity -> do
    let filePath = dataDir </> show indexNum </> ("pos_org_relations_" ++ show arity ++ ".csv")
    csvLines <- readCsv filePath
    return $ parseRelations arity csvLines
  let posOrgRelations' = Map.unionsWith (++) posOrgRelationsList :: Map.Map Int [([Int], Int)]
  
  posAddRelationsList <- forM arities $ \arity -> do
    let filePath = dataDir </> show indexNum </> ("pos_add_relations_" ++ show arity ++ ".csv")
    csvLines <- readCsv filePath
    return $ parseRelations arity csvLines
  let posAddRelations' = Map.unionsWith (++) posAddRelationsList :: Map.Map Int [([Int], Int)]
  
  putStrLn "posOrgRelations Sizes:"
  mapM_ (\(arity, rels) -> putStrLn $ "Arity " ++ show arity ++ ": " ++ show (length rels)) (Map.toList posOrgRelations')
  putStrLn "posAddRelations Sizes:"
  mapM_ (\(arity, rels) -> putStrLn $ "Arity " ++ show arity ++ ": " ++ show (length rels)) (Map.toList posAddRelations')
  ----
  
  -- 複数回学習を行い、平均精度を出力
  forM_ arities $ \arity -> do
    ---- ファイルから読み込み
    entityDict <- readEntityDict (dataDir </> show indexNum </> "entity_dict_" ++ show arity ++ ".csv")
    let posOrgRelations = Map.map (\relations -> groupBySentence relations entityDict) posOrgRelations' :: Map.Map Int [[([Int], Int)]]
    let posAddRelations = Map.map (\relations -> groupBySentence relations entityDict) posAddRelations' :: Map.Map Int [[([Int], Int)]]
    ----

    ---- 学習とテスト
    let posOrgRelationsForArity = Map.findWithDefault [] arity posOrgRelations -- :: [[([Int], Int)]]
    let posAddRelationsForArity = Map.findWithDefault [] arity posAddRelations -- :: [[([Int], Int)]]
    -- scores <- forM [1..2] $ \i -> trainAndTest i arity posOrgRelationsForArity posAddRelationsForArity
    scores <- forM [1..2] $ \i -> trainAndTest2 i arity posOrgRelationsForArity posAddRelationsForArity
    let (accuracies, precisions, recalls, f1Scores) = L.unzip4 scores
    let averageAccuracy = sum accuracies / fromIntegral (length accuracies)
    let averagePrecision = sum precisions / fromIntegral (length precisions)
    let averageRecall = sum recalls / fromIntegral (length recalls)
    let averageF1Score = sum f1Scores / fromIntegral (length f1Scores)
    putStrLn $ "Average Accuracy for arity " ++ show arity ++ ": " ++ show averageAccuracy
    putStrLn $ "Average Precision for arity " ++ show arity ++ ": " ++ show averagePrecision
    putStrLn $ "Average Recall for arity " ++ show arity ++ ": " ++ show averageRecall
    putStrLn $ "Average F1 Score for arity " ++ show arity ++ ": " ++ show averageF1Score
    
    ---- テストのみ
    -- accuracy <- testOnly 1 arity posOrgRelationsForArity posAddRelationsForArity
    -- putStrLn $ "Accuracy for arity " ++ show arity ++ ": " ++ show accuracy

trainAndTest2 :: Int -> Int -> [[([Int], Int)]] -> [[([Int], Int)]] -> IO (Double, Double, Double, Double)
trainAndTest2 fold arity posOrgRelations posAddRelations = do
  -- 1. 各意味表示からtest候補を選ぶ．残りがpos training data
  let (trainPosData, testPosData) = unzip $ map splitRelations posOrgRelations
  let flatTrainPosData = concat trainPosData :: [([Int], Int)]
  let flatTestPosData = concat testPosData :: [([Int], Int)]
  putStrLn $ "Train Data Sizes: " ++ show (length flatTrainPosData)
  putStrLn $ "Test PosData Sizes: " ++ show (length flatTestPosData)
  
  -- 2. エンティティ辞書を読み込み、エンティティを変換
  entityDict <- readEntityDict (dataDir </> show indexNum </> "entity_dict_" ++ show arity ++ ".csv") -- :: Map.Map Int String
  let parsedEntityDict = Map.map parseElement entityDict
  let convertEntity entity = Map.findWithDefault (entity, []) entity parsedEntityDict

  -- 空のπシーケンスをフィルタリングする関数
  let filterEmptyPi entities = 
        let filtered = filter (\(_, piSeq) -> null piSeq) entities
        in if not (null filtered)
           then trace ("Filtered entities: " ++ show filtered) False
           else True

  -- エンティティを変換し、フィルタリングを行う
  let convertAndFilter (xs, y) = 
        let converted = map convertEntity xs
        in if filterEmptyPi converted then Just (converted, y) else Nothing

  let filteredTrainPosData = mapMaybe convertAndFilter flatTrainPosData :: [([(Int, [Int])], Int)]
      filteredTestPosData = mapMaybe convertAndFilter flatTestPosData :: [([(Int, [Int])], Int)]

  putStrLn $ "Filtered Train Data Sizes: " ++ show (length filteredTrainPosData)
  putStrLn $ "Filtered Test Data Sizes: " ++ show (length filteredTestPosData)

  shuffledTrainPosData <- shuffleM filteredTrainPosData -- no add 
  shuffledTestPosData <- shuffleM filteredTestPosData

  -- 3. 2.に含まれないデータをneg dataとして生成
  let allEntities = concatMap fst flatTrainPosData -- no add :: [Int]
  let allPreds = map snd flatTrainPosData -- no add
  let existingRelations = Set.fromList (shuffledTrainPosData ++ shuffledTestPosData)
  negData <- generateNegRelations4 shuffledTrainPosData allPreds existingRelations (length shuffledTrainPosData + length shuffledTestPosData)

  -- 4. 3.をneg train dataとneg test dataに分割する
  let (trainNegData, testNegData) = splitAt (length shuffledTrainPosData) negData
  putStrLn $ "Train NegData Sizes:" ++ show (length trainNegData)
  putStrLn $ "Test NegData Sizes:" ++ show (length testNegData)

  shuffledTrainNegData <- shuffleM trainNegData
  shuffledTestNegData <- shuffleM testNegData

  writeRelationsCsv2 (dataDir </> show indexNum </> "train_pos_" ++ show arity ++ ".csv") shuffledTrainPosData
  writeRelationsCsv2 (dataDir </> show indexNum </> "train_neg_" ++ show arity ++ ".csv") shuffledTrainNegData
  writeRelationsCsv2 (dataDir </> show indexNum </> "test_pos_" ++ show arity ++ ".csv") shuffledTestPosData
  writeRelationsCsv2 (dataDir </> show indexNum </> "test_neg_" ++ show arity ++ ".csv") shuffledTestNegData

  -- 5. 追加した pos training data と neg training dataがtraining data、1.のpos test dataと4.のneg test dataがtest data
  let trainPosData' = map (\(xs, y) -> ((xs, y), 1.0)) shuffledTrainPosData :: [(([(Int, [Int])], Int), Float)]
  let trainNegData' = map (\(xs, y) -> ((xs, y), 0.0)) shuffledTrainNegData
  let trainData = trainPosData' ++ trainNegData' :: [(([(Int, [Int])], Int), Float)]
  let testPosData' = map (\(xs, y) -> ((xs, y), 1.0)) shuffledTestPosData
  let testNegData' = map (\(xs, y) -> ((xs, y), 0.0)) shuffledTestNegData
  let testData = testPosData' ++ testNegData' :: [(([(Int, [Int])], Int), Float)]

  genTrain <- newStdGen
  genTest <- newStdGen
  let shuffledTrainData = shuffle' trainData (length trainData) genTrain
  let shuffledTestData = shuffle' testData (length testData) genTest

  let entities1 = concatMap fst filteredTrainPosData
  let entities2 = concatMap fst filteredTestPosData
  let uniqueEntities = Set.toList $ Set.fromList (entities1 ++ entities2)
  putStrLn $ "Unique entities: " ++ show (length uniqueEntities)

  let entityCount = length uniqueEntities
  -- entityCount <- Utils.getLineCount (dataDir </> show indexNum </> "entity_dict_" ++ show arity ++ ".csv")
  relationCount <- Utils.getLineCount (dataDir </> show indexNum </> "predicate_dict_" ++ show arity ++ ".csv")
  putStrLn $ "entityCount: " ++ show entityCount
  putStrLn $ "relationCount: " ++ show relationCount
  S.hFlush S.stdout

  -- MLPを使用する場合
  let mlpSpec = MLP2.MLPSpec {
            entity_num_embed = entityCount,
            relation_num_embed = relationCount,
            entity_features = 256,
            relation_features = 256,
            hidden_dim1 = 256,
            hidden_dim2 = 32,
            output_feature = 1,
            arity = arity}
  let modelName = "MLP_arity_" ++ show arity ++ "_fold_" ++ show fold
  MLP2.trainModel modelName mlpSpec shuffledTrainData arity
  (accuracy, precision, recall, f1Score) <- MLP2.testModel modelName mlpSpec shuffledTestData arity
  return (accuracy, precision, recall, f1Score)
  -- return (0.0, 0.0, 0.0, 0.0)

parseElement :: String -> (Int, [Int])
parseElement str = 
  let regex = "S([0-9]+)" :: String
      piRegex = "π([12])" :: String
      sMatch = str =~ regex :: [[String]]
      piMatches = str =~ piRegex :: [[String]]
      sNumber = if not (null sMatch) then read (sMatch !! 0 !! 1) :: Int else 0
      piSequence = map (\m -> read (m !! 1) :: Int) piMatches
  in (sNumber, piSequence)

writeRelationsCsv2 :: FilePath -> [([(Int, [Int])], Int)] -> IO ()
writeRelationsCsv2 path relations = S.withFile path S.WriteMode $ \h -> do
  let formatEntity (entity, piSeq) = show entity ++ ":" ++ L.intercalate "-" (map show piSeq)
  let formatRelation (entities, p) = L.intercalate "," (map formatEntity entities ++ [show p])
  let content = unlines $ map formatRelation relations
  S.hPutStr h content
  
trainAndTest :: Int -> Int -> [[([Int], Int)]] -> [[([Int], Int)]] -> IO (Double, Double, Double, Double)
trainAndTest fold arity posOrgRelations posAddRelations = do
  -- 1. 各意味表示からtest候補を選ぶ．残りがpos training data
  let (trainPosData, testPosData) = unzip $ map splitRelations posOrgRelations
  let flatTrainPosData = concat trainPosData :: [([Int], Int)]
  let flatTestPosData = concat testPosData :: [([Int], Int)]

  -- 2. pos training dataを追加する
  let flatPosAddData = concat posAddRelations :: [([Int], Int)]
  let addedTrainPosData = flatTrainPosData ++ flatPosAddData :: [([Int], Int)]
  putStrLn $ "Train Data Sizes: " ++ show (length flatTrainPosData) ++ "->" ++ show (length addedTrainPosData)
  putStrLn $ "Test PosData Sizes: " ++ show (length flatTestPosData)
  
  shuffledTrainPosData <- shuffleM addedTrainPosData
  -- shuffledTrainPosData <- shuffleM flatTrainPosData -- no add 
  shuffledTestPosData <- shuffleM flatTestPosData

  -- 3. 2.に含まれないデータをneg dataとして生成
  let allEntities = concatMap fst addedTrainPosData
  let allPreds = map snd addedTrainPosData
  -- let allEntities = concatMap fst flatTrainPosData -- no add
  -- let allPreds = map snd flatTrainPosData -- no add
  let existingRelations = Set.fromList (shuffledTrainPosData ++ shuffledTestPosData)
  -- negData <- generateNegRelations2 shuffledTrainPosData allPreds existingRelations (length shuffledTrainPosData + length flatTestPosData)
  negData <- generateNegRelations2 shuffledTrainPosData allEntities allPreds existingRelations (length shuffledTrainPosData + length shuffledTestPosData)

  -- 4. 3.をneg train dataとneg test dataに分割する
  let (trainNegData, testNegData) = splitAt (length shuffledTrainPosData) negData
  putStrLn $ "Train NegData Sizes:" ++ show (length trainNegData)
  putStrLn $ "Test NegData Sizes:" ++ show (length testNegData)

  shuffledTrainNegData <- shuffleM trainNegData
  shuffledTestNegData <- shuffleM testNegData

  writeRelationsCsv (dataDir </> show indexNum </> "train_pos_" ++ show arity ++ ".csv") shuffledTrainPosData
  writeRelationsCsv (dataDir </> show indexNum </> "train_neg_" ++ show arity ++ ".csv") shuffledTrainNegData
  writeRelationsCsv (dataDir </> show indexNum </> "test_pos_" ++ show arity ++ ".csv") shuffledTestPosData
  writeRelationsCsv (dataDir </> show indexNum </> "test_neg_" ++ show arity ++ ".csv") shuffledTestNegData

  -- 5. 追加した pos training data と neg training dataがtraining data、1.のpos test dataと4.のneg test dataがtest data
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
            hidden_dim1 = 256,
            hidden_dim2 = 32,
            output_feature = 1,
            arity = arity}
  let modelName = "MLP_arity_" ++ show arity ++ "_fold_" ++ show fold
  MLP.trainModel modelName mlpSpec shuffledTrainData arity
  (accuracy, precision, recall, f1Score) <- MLP.testModel modelName mlpSpec shuffledTestData arity

  -- Socher NTNを使用する場合 (TODO : n=2の場合以外も対応する)
  -- let sntnSpec = SNTN.SNTNSpec { 
  --   entity_num_embed = entityCount, 
  --   relation_num_embed = relationCount, 
  --   embedding_features = 128, 
  --   output_dim = 1 }
  -- let modelName = "SNTN_arity_" ++ show arity ++ "_fold_" ++ show fold
  -- SNTN.trainModel modelName sntnSpec shuffledTrainData arity
  -- (accuracy, precision, recall, f1Score) <- SNTN.testModel modelName sntnSpec shuffledTestData arity

  -- Ding NTNを使用する場合 (TODO : n=2の場合以外も対応する)
  -- let dntnSpec = DNTN.DNTNSpec { 
  --   entity_num_embed = entityCount, 
  --   relation_num_embed = relationCount, 
  --   embedding_features = 256,
  --   tensor_dim = 256,
  --   num_arguments = arity,
  --   dropout_probability = 0.1 }
  -- let modelName = "DNTN_arity_" ++ show arity ++ "_fold_" ++ show fold
  -- DNTN.trainModel modelName dntnSpec shuffledTrainData arity
  -- (accuracy, precision, recall, f1Score) <- DNTN.testModel modelName dntnSpec shuffledTestData arity

  return (accuracy, precision, recall, f1Score)

testOnly :: Int -> Int -> [[([Int], Int)]] -> [[([Int], Int)]] -> IO Double
testOnly fold arity posOrgRelations posAddRelations = do
  testPosData <- readRelationsCsv (dataDir </> show indexNum </> "test_pos_" ++ show arity ++ ".csv") 
  testNegData <- readRelationsCsv (dataDir </> show indexNum </> "test_neg_" ++ show arity ++ ".csv") 

  let testPosData' = map (\(xs, y) -> ((xs, y), 1.0)) testPosData
  let testNegData' = map (\(xs, y) -> ((xs, y), 0.0)) testNegData
  let testData = testPosData' ++ testNegData' :: [(([Int], Int), Float)]

  genTest <- newStdGen
  let shuffledTestData = shuffle' testData (length testData) genTest

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
            arity = arity}
  let modelName = "MLP_arity_" ++ show arity ++ "_fold_" ++ show fold
  (accuracy, precision, recall, f1Score) <- MLP.testModel modelName mlpSpec shuffledTestData arity

  -- Socher NTNを使用する場合 (TODO : n=2の場合以外も対応する)
  -- let sntnSpec = SNTN.SNTNSpec { 
  --   entity_num_embed = entityCount, 
  --   relation_num_embed = relationCount, 
  --   embedding_features = 128, 
  --   output_dim = 1 }
  -- let modelName = "SNTN_arity_" ++ show arity ++ "_fold_" ++ show fold
  -- (accuracy, precision, recall, f1Score) <- SNTN.testModel modelName sntnSpec shuffledTestData arity

  -- Ding NTNを使用する場合 (TODO : n=2の場合以外も対応する)
  -- let dntnSpec = DNTN.DNTNSpec { 
  --   entity_num_embed = entityCount, 
  --   relation_num_embed = relationCount, 
  --   embedding_features = 256,
  --   tensor_dim = 256,
  --   num_arguments = arity,
  --   dropout_probability = 0.1 }
  -- let modelName = "DNTN_arity_" ++ show arity ++ "_fold_" ++ show fold
  -- (accuracy, precision, recall, f1Score) <- DNTN.testModel modelName dntnSpec shuffledTestData arity

  putStrLn $ "Accuracy: " ++ show accuracy
  return accuracy

-- CSVファイルを読み込む関数
readCsv :: FilePath -> IO [T.Text]
readCsv path = do
  content <- S.readFile path
  return $ T.lines (T.pack content)

readRelationsCsv :: FilePath -> IO [([Int], Int)]
readRelationsCsv path = do
  content <- S.readFile path
  let lines = T.lines (T.pack content)
  return $ map parseLine lines
  where
    parseLine line = 
      let fields = map (read . T.unpack) (T.splitOn "," line)
      in (init fields, last fields)

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
generateNegRelations4 :: [([(Int, [Int])], Int)] -> [Int] -> Set.Set ([(Int, [Int])], Int) -> Int -> IO [([(Int, [Int])], Int)]
generateNegRelations4 posRelations allPreds existingNegRelations numNegRelations = do
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

generateNegRelations3 :: [([Int], Int)] -> [Int] -> Set.Set ([Int], Int) -> Int -> IO [([Int], Int)]
generateNegRelations3 posRelations allPreds existingNegRelations numNegRelations = do
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

generateNegRelations2 :: [([Int], Int)] -> [Int] -> [Int] -> Set.Set ([Int], Int) -> Int -> IO [([Int], Int)]
generateNegRelations2 posRelations allEntities allPreds existingNegRelations numNegRelations = do
  let posSet = Set.fromList posRelations
  let generateOneNegRelation = do
        -- ランダムにエンティティを選ぶ
        entities <- replicateM (length (fst (head posRelations))) $ randomRIO (0, length allEntities - 1) >>= \i -> return (allEntities !! i)
        -- ランダムに述語を選ぶ
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
