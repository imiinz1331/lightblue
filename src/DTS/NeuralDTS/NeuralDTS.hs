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
import System.Directory (createDirectoryIfMissing)
import Debug.Trace

import DTS.NeuralDTS.PreProcess (getTrainRelations)
import DTS.NeuralDTS.Classifier.MLP (trainModel, testModel, crossValidation)
import qualified Parser.ChartParser as CP
import Parser.Language (jpOptions) 
import qualified Parser.Language.Japanese.Juman.CallJuman as Juman
import qualified Parser.Language.Japanese.Lexicon as L (LexicalResource(..), lexicalResourceBuilder, LexicalItems, lookupLexicon, setupLexicon, emptyCategories, myLexicon)
import Control.Monad.RWS (MonadState(put))

inputsDir = "src/DTS/NeuralDTS/inputs"
dataSetDir = "src/DTS/NeuralDTS/dataSet"
imagesDir = "src/DTS/NeuralDTS/images"
modelsDir = "src/DTS/NeuralDTS/models"
indexNum = 12

checkAccuracy :: CP.ParseSetting -> [T.Text] -> IO ()
checkAccuracy ps str = do
  createDirectoryIfMissing True (dataSetDir </> show indexNum)
  createDirectoryIfMissing True (imagesDir </> show indexNum)
  createDirectoryIfMissing True (modelsDir </> show indexNum)
  -- ((posOrgRelations, posAddRelations), (negOrgRelations, negAddRelations)) <- getTrainRelations ps str -- :: (Map.Map Int [([Int], Int)], Map.Map Int [([Int], Int)])

  -- putStrLn "Training Relations:"
  -- print posRelations
  -- print negRelations

  -- ファイルから読み込み
  let arities = [2]
  posOrgRelationsList <- forM arities $ \arity -> do
    let filePath = dataSetDir </> show indexNum </> ("pos_org_relations_" ++ show arity ++ ".csv")
    csvLines <- readCsv filePath
    return $ parseRelations arity csvLines
  let posOrgRelations = Map.unionsWith (++) posOrgRelationsList
  
  posAddRelationsList <- forM arities $ \arity -> do
    let filePath = dataSetDir </> show indexNum </> ("pos_add_relations_" ++ show arity ++ ".csv")
    csvLines <- readCsv filePath
    return $ parseRelations arity csvLines
  let posAddRelations = Map.unionsWith (++) posAddRelationsList

  negOrgRelationsList <- forM arities $ \arity -> do
    let filePath = dataSetDir </> show indexNum </> ("neg_org_relations_" ++ show arity ++ ".csv")
    csvLines <- readCsv filePath
    return $ parseRelations arity csvLines
  let negOrgRelations = Map.unionsWith (++) negOrgRelationsList

  negAddRelationsList <- forM arities $ \arity -> do
    let filePath = dataSetDir </> show indexNum </> ("neg_add_relations_" ++ show arity ++ ".csv")
    csvLines <- readCsv filePath
    return $ parseRelations arity csvLines
  let negAddRelations = Map.unionsWith (++) negAddRelationsList

  -- ラベルづけ
  let posOrgRelations' = Map.map (map (\(xs, y) -> ((xs, y), 1.0))) posOrgRelations
  let posAddRelations' = Map.map (map (\(xs, y) -> ((xs, y), 1.0))) posAddRelations
  let negOrgRelations' = Map.map (map (\(xs, y) -> ((xs, y), 0.0))) negOrgRelations
  let negAddRelations' = Map.map (map (\(xs, y) -> ((xs, y), 0.0))) negAddRelations

  let posOrgRelationsList2 = concatMap (\(k, v) -> map (\x -> (k, x)) v) (Map.toList posOrgRelations') :: [(Int, (([Int], Int), Float))]
  let posAddRelationsList2 = concatMap (\(k, v) -> map (\x -> (k, x)) v) (Map.toList posAddRelations') :: [(Int, (([Int], Int), Float))]
  let negOrgRelationsList2 = concatMap (\(k, v) -> map (\x -> (k, x)) v) (Map.toList negOrgRelations') :: [(Int, (([Int], Int), Float))]
  let negAddRelationsList2 = concatMap (\(k, v) -> map (\x -> (k, x)) v) (Map.toList negAddRelations') :: [(Int, (([Int], Int), Float))]

  let orgData = posOrgRelationsList2 ++ negOrgRelationsList2
  let addData = posAddRelationsList2 ++ negAddRelationsList2

  genOrg <- newStdGen
  genAdd <- newStdGen
  let shuffledOrgData = shuffle' orgData (length orgData) genOrg
  let shuffledAddData = shuffle' addData (length addData) genAdd

  putStrLn $ "TrainValid Relations Count: " ++ show (length orgData)
  S.hFlush S.stdout

  let orgDataMap = Map.fromListWith (++) [(k, [v]) | (k, v) <- shuffledOrgData]
  let addDataMap = Map.fromListWith (++) [(k, [v]) | (k, v) <- shuffledAddData]
  
  mapM_ (\arity -> do
          let orgDataForArity = Map.findWithDefault [] arity orgDataMap
          let addDataForArity = Map.findWithDefault [] arity addDataMap
          averageAccuracy <- crossValidation 5 orgDataForArity addDataForArity arity
          putStrLn $ "Average accuracy for arity " ++ show arity ++ ": " ++ show averageAccuracy ++ "%"
        ) (Map.keys orgDataMap)

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
  let posStr2 = take 2000 posStr

  lr <- L.lexicalResourceBuilder Juman.KWJA
  let ps = CP.ParseSetting jpOptions lr 1 1 1 1 True Nothing Nothing True False

  -- トレーニングとテストを実行
  checkAccuracy ps posStr2