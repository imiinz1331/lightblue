{-# LANGUAGE DeriveAnyClass #-}
-- {-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module DTS.NeuralDTS.NeuralDTS  (
  testProcessAndTrain,
  testNeuralDTS
  ) where

import Control.Monad (forM, replicateM)
import qualified Data.Text.Lazy as T      --text
import qualified Data.Text.Lazy.IO as T   --text
import qualified Data.List as L           --base
import qualified Data.Map as Map
import qualified Data.Set as Set
import qualified Interface.Text as T
import qualified System.IO as S
import System.FilePath ((</>))
import System.Random (randomRIO, newStdGen, randomRs)
import System.Random.Shuffle (shuffle, shuffle')
import Debug.Trace

import DTS.NeuralDTS.PreProcess (extractPredicateName, getTrainRelations)
import DTS.NeuralDTS.Classifier.MLP (trainModel, testModel)
import qualified Parser.ChartParser as CP
import Parser.Language (jpOptions) 
import qualified Parser.Language.Japanese.Juman.CallJuman as Juman
import qualified Parser.Language.Japanese.Lexicon as L (LexicalResource(..), lexicalResourceBuilder, LexicalItems, lookupLexicon, setupLexicon, emptyCategories, myLexicon)
import Control.Monad.RWS (MonadState(put))

inputsDir = "src/DTS/NeuralDTS/inputs"
dataSetDir = "src/DTS/NeuralDTS/dataSet"
indexNum = 12

-- processAndTrain :: CP.ParseSetting -> [T.Text] -> IO ()
-- processAndTrain ps posStr = do
--   (posTrainRelations, negTrainRelations) <- getTrainRelations ps posStr -- :: (Map.Map Int [([Int], Int)], Map.Map Int [([Int], Int)])
--   putStrLn "Training Relations:"
--   -- print posTrainRelations
--   -- print negTrainRelations

--   let modelName = "mlp-model" ++ show indexNum
--   let posTrainRelations' = Map.map (map (\(xs, y) -> ((xs, y), 1.0))) posTrainRelations
--   let negTrainRelations' = Map.map (map (\(xs, y) -> ((xs, y), 0.0))) negTrainRelations
--   let allTrainRelations = Map.unionWith (++) posTrainRelations' negTrainRelations'
  
--   -- n項関係のモデルをトレーニング
--   mapM_ (\(arity, relations) -> trainModel modelName relations arity) (Map.toList allTrainRelations)

-- processAndTest :: CP.ParseSetting -> [T.Text] -> IO ()
-- processAndTest ps str = do
--   testRelationsByArity <- getTestRelations ps str -- :: Map.Map Int [([Int], Int)]
--   putStrLn "Test Relations:"
--   print testRelationsByArity

--   -- MLP モデルのロードとテスト
--   let modelName = "mlp-model3"
--   mapM_ (\(arity, relations) -> testModel modelName relations arity) (Map.toList testRelationsByArity)

checkAccuracy :: CP.ParseSetting -> [T.Text] -> IO ()
checkAccuracy ps str = do
  ((posOrgRelations, posAddRelations), (negOrgRelations, negAddRelations)) <- getTrainRelations ps str -- :: (Map.Map Int [([Int], Int)], Map.Map Int [([Int], Int)])

  -- putStrLn "Training Relations:"
  -- print posRelations
  -- print negRelations

  -- ファイルから読み込み
  -- let arities = [2]
  -- posOrgRelationsList <- forM arities $ \arity -> do
  --   let filePath = dataSetDir </> ("pos_org_relations_" ++ show arity ++ "_" ++ show indexNum ++ ".csv")
  --   csvLines <- readCsv filePath
  --   return $ parseRelations arity csvLines
  -- let posOrgRelations = Map.unionsWith (++) posOrgRelationsList
  
  -- posAddRelationsList <- forM arities $ \arity -> do
  --   let filePath = dataSetDir </> ("pos_add_relations_" ++ show arity ++ "_" ++ show indexNum ++ ".csv")
  --   csvLines <- readCsv filePath
  --   return $ parseRelations arity csvLines
  -- let posAddRelations = Map.unionsWith (++) posAddRelationsList

  -- negOrgRelationsList <- forM arities $ \arity -> do
  --   let filePath = dataSetDir </> ("neg_org_relations_" ++ show arity ++ "_" ++ show indexNum ++ ".csv")
  --   csvLines <- readCsv filePath
  --   return $ parseRelations arity csvLines
  -- let negOrgRelations = Map.unionsWith (++) negOrgRelationsList

  -- negAddRelationsList <- forM arities $ \arity -> do
  --   let filePath = dataSetDir </> ("neg_add_relations_" ++ show arity ++ "_" ++ show indexNum ++ ".csv")
  --   csvLines <- readCsv filePath
  --   return $ parseRelations arity csvLines
  -- let negAddRelations = Map.unionsWith (++) negAddRelationsList

  -- ラベルづけ
  let posOrgRelations' = Map.map (map (\(xs, y) -> ((xs, y), 1.0))) posOrgRelations
  let posAddRelations' = Map.map (map (\(xs, y) -> ((xs, y), 1.0))) posAddRelations
  let negOrgRelations' = Map.map (map (\(xs, y) -> ((xs, y), 0.0))) negOrgRelations
  let negAddRelations' = Map.map (map (\(xs, y) -> ((xs, y), 0.0))) negAddRelations

  let posOrgRelationsList2 = concatMap (\(k, v) -> map (\x -> (k, x)) v) (Map.toList posOrgRelations') :: [(Int, (([Int], Int), Float))]
  let posAddRelationsList2 = concatMap (\(k, v) -> map (\x -> (k, x)) v) (Map.toList posAddRelations') :: [(Int, (([Int], Int), Float))]
  let negOrgRelationsList2 = concatMap (\(k, v) -> map (\x -> (k, x)) v) (Map.toList negOrgRelations') :: [(Int, (([Int], Int), Float))]
  let negAddRelationsList2 = concatMap (\(k, v) -> map (\x -> (k, x)) v) (Map.toList negAddRelations') :: [(Int, (([Int], Int), Float))]

  genPos <- newStdGen
  genNeg <- newStdGen
  let shuffledOrgPosList = shuffle' posOrgRelationsList2 (length posOrgRelationsList2) genPos
  let shuffledOrgNegList = shuffle' negOrgRelationsList2 (length negOrgRelationsList2) genNeg

  -- データを分割
  let (trainPosData, restPosData) = splitAt (round $ 0.5 * fromIntegral (length shuffledOrgPosList)) shuffledOrgPosList
  let (validPosData, testPosData) = splitAt (round $ 0.5 * fromIntegral (length restPosData)) restPosData
  let (trainNegData, restNegData) = splitAt (round $ 0.5 * fromIntegral (length shuffledOrgNegList)) shuffledOrgNegList
  let (validNegData, testNegData) = splitAt (round $ 0.5 * fromIntegral (length restNegData)) restNegData
  let trainAddPosData = trainPosData ++ posAddRelationsList2
  let trainAddNegData = trainNegData ++ negAddRelationsList2

  genPos' <- newStdGen
  genNeg' <- newStdGen
  let trainPosData' = shuffle' trainAddPosData (length trainAddPosData) genPos'
  let trainNegData' = shuffle' trainAddNegData (length trainAddNegData) genNeg'

  writeCsv (dataSetDir </> "trainPosData.csv") (map (\(arity, ((entities, pred), _)) -> (show arity ++ "," ++ show entities, pred)) trainPosData')
  writeCsv (dataSetDir </> "validPosData.csv") (map (\(arity, ((entities, pred), _)) -> (show arity ++ "," ++ show entities, pred)) validPosData)
  writeCsv (dataSetDir </> "testPosData.csv") (map (\(arity, ((entities, pred), _)) -> (show arity ++ "," ++ show entities, pred)) testPosData)
  writeCsv (dataSetDir </> "trainNegData.csv") (map (\(arity, ((entities, pred), _)) -> (show arity ++ "," ++ show entities, pred)) trainNegData')
  writeCsv (dataSetDir </> "validNegData.csv") (map (\(arity, ((entities, pred), _)) -> (show arity ++ "," ++ show entities, pred)) validNegData)
  writeCsv (dataSetDir </> "testNegData.csv") (map (\(arity, ((entities, pred), _)) -> (show arity ++ "," ++ show entities, pred)) testNegData)

  let trainData = trainPosData' ++ trainNegData'
  let validData = validPosData ++ validNegData
  let testData = testPosData ++ testNegData

  genTrain <- newStdGen
  genValid <- newStdGen
  genTest <- newStdGen
  let shuffledTrainData = shuffle' trainData (length trainData) genTrain
  let shuffledValidData = shuffle' validData (length validData) genValid
  let shuffledTestData = shuffle' testData (length testData) genTest

  writeCsv (dataSetDir </> "trainData.csv") (map (\(arity, ((entities, pred), label)) -> (show arity ++ "," ++ show label ++ ":" ++ show entities, pred)) shuffledTrainData)
  writeCsv (dataSetDir </> "validData.csv") (map (\(arity, ((entities, pred), label)) -> (show arity ++ "," ++ show label ++ ":" ++ show entities, pred)) shuffledValidData)
  writeCsv (dataSetDir </> "testData.csv") (map (\(arity, ((entities, pred), label)) -> (show arity ++ "," ++ show label ++ ":" ++ show entities, pred)) shuffledTestData)

  putStrLn $ "Train Relations Count: " ++ show (length trainData)
  putStrLn $ "Valid Relations Count: " ++ show (length validData)
  putStrLn $ "Test Relations Count: " ++ show (length testData)
  S.hFlush S.stdout

  -- 分割したデータをMapに戻す
  let trainDataMap = Map.fromListWith (++) [(k, [v]) | (k, v) <- shuffledTrainData]
  let validDataMap = Map.fromListWith (++) [(k, [v]) | (k, v) <- shuffledValidData]
  let testDataMap = Map.fromListWith (++) [(k, [v]) | (k, v) <- shuffledTestData]
  
  let modelName = "mlp-model" ++ show indexNum
  mapM_ (\arity -> do
           let trainDataForArity = Map.findWithDefault [] arity trainDataMap
           let validDataForArity = Map.findWithDefault [] arity validDataMap
           trainModel modelName trainDataForArity validDataForArity arity
        ) (Map.keys trainDataMap)

  mapM_ (\arity -> do
           let testDataForArity = Map.findWithDefault [] arity testDataMap
           testModel modelName testDataForArity arity
        ) (Map.keys testDataMap)

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

parseRelations :: Int -> [T.Text] -> Map.Map Int [([Int], Int)]
parseRelations arity lines =
  Map.fromListWith (++) [(arity, [(init entities, last entities)]) | line <- lines, let entities = map (read . T.unpack) (T.splitOn (T.pack ",") line) :: [Int]]

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