{-# LANGUAGE DeriveAnyClass #-}
-- {-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module DTS.NeuralDTS.NeuralDTS  (
  processAndTrain,
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

import DTS.NeuralDTS.PreProcess (extractPredicateName, getTrainRelations, getTestRelations)
import DTS.NeuralDTS.Classifier.MLP (trainModel, testModel)
import qualified Parser.ChartParser as CP
import Parser.Language (jpOptions) 
import qualified Parser.Language.Japanese.Juman.CallJuman as Juman
import qualified Parser.Language.Japanese.Lexicon as L (LexicalResource(..), lexicalResourceBuilder, LexicalItems, lookupLexicon, setupLexicon, emptyCategories, myLexicon)

inputsDir = "src/DTS/NeuralDTS/inputs"
dataSetDir = "src/DTS/NeuralDTS/dataSet"
indexNum = 7

processAndTrain :: CP.ParseSetting -> [T.Text] -> IO ()
processAndTrain ps posStr = do
  (posTrainRelations, negTrainRelations) <- getTrainRelations ps posStr -- :: (Map.Map Int [([Int], Int)], Map.Map Int [([Int], Int)])
  putStrLn "Training Relations:"
  -- print posTrainRelations
  -- print negTrainRelations

  let modelName = "mlp-model" ++ show indexNum
  let posTrainRelations' = Map.map (map (\(xs, y) -> ((xs, y), 1.0))) posTrainRelations
  let negTrainRelations' = Map.map (map (\(xs, y) -> ((xs, y), 0.0))) negTrainRelations
  let allTrainRelations = Map.unionWith (++) posTrainRelations' negTrainRelations'
  
  -- n項関係のモデルをトレーニング
  mapM_ (\(arity, relations) -> trainModel modelName relations arity) (Map.toList allTrainRelations)

-- processAndTest :: CP.ParseSetting -> [T.Text] -> IO ()
-- processAndTest ps str = do
--   testRelationsByArity <- getTestRelations ps str -- :: Map.Map Int [([Int], Int)]
--   putStrLn "Test Relations:"
--   print testRelationsByArity

--   -- MLP モデルのロードとテスト
--   let modelName = "mlp-model3"
--   mapM_ (\(arity, relations) -> testModel modelName relations arity) (Map.toList testRelationsByArity)

remapEntitiesAndPredicates :: Map.Map Int [([Int], Int)] -> (Map.Map Int Int, Map.Map Int Int, Map.Map Int [([Int], Int)])
remapEntitiesAndPredicates posRelations = 
  let usedEntities = Set.fromList [entity | (_, posRelations) <- Map.toList posRelations, (entities, _) <- posRelations, entity <- entities]
      usedPredicates = Set.fromList [pred | (_, posRelations) <- Map.toList posRelations, (_, pred) <- posRelations]
      entityList = Set.toList usedEntities
      predicateList = Set.toList usedPredicates
      newEntityMap = Map.fromList (zip entityList [0..])
      newPredicateMap = Map.fromList (zip predicateList [0..])
      remapEntity entity = Map.findWithDefault entity entity newEntityMap
      remapPredicate pred = Map.findWithDefault pred pred newPredicateMap
      remapRelation (entities, pred) = (map remapEntity entities, remapPredicate pred)
      newPosRelations = Map.map (map remapRelation) posRelations
  in (newEntityMap, newPredicateMap, newPosRelations)

checkAccuracy :: CP.ParseSetting -> [T.Text] -> IO ()
checkAccuracy ps str = do
  -- (posRelations, negRelations) <- getTrainRelations ps str -- :: (Map.Map Int [([Int], Int)], Map.Map Int [([Int], Int)])

  -- ファイルから読み込み
  let arities = [2]
  posRelationsList <- forM arities $ \arity -> do
    let filePath = dataSetDir </> ("new_pos_relations_arity" ++ show arity ++ "_" ++ show indexNum ++ ".csv")
    csvLines <- readCsv filePath
    return $ parseRelations arity csvLines
  let posRelations = Map.unionsWith (++) posRelationsList

  negRelationsList <- forM arities $ \arity -> do
    let filePath = dataSetDir </> ("new_neg_relations_arity" ++ show arity ++ "_" ++ show indexNum ++ ".csv")
    csvLines <- readCsv filePath
    return $ parseRelations arity csvLines
  let negRelations = Map.unionsWith (++) negRelationsList

  -- -- エンティティと述語をリマップ
  -- forM arities $ \arity -> do
  --   let arityPosRelations = Map.filterWithKey (\k _ -> k == arity) posRelations
  --   let newPosRelationsList = concatMap (\(k, v) -> map (\(entities, pred) -> (show k ++ "," ++ show entities, pred)) v) (Map.toList arityPosRelations)
  --   writeCsv (dataSetDir </> "new_pos_relations_arity" ++ show arity ++ "_" ++ show indexNum ++ ".csv") newPosRelationsList

  -- forM arities $ \arity -> do
  --   let arityNegRelations = Map.filterWithKey (\k _ -> k == arity) posRelations
  --   let newNegRelationsList = concatMap (\(k, v) -> map (\(entities, pred) -> (show k ++ "," ++ show entities, pred)) v) (Map.toList arityNegRelations)
  --   writeCsv (dataSetDir </> "new_neg_relations_arity" ++ show arity ++ "_" ++ show indexNum ++ ".csv") newNegRelationsList

  -- let newPosRelationsList = concatMap (\(k, v) -> map (\(entities, pred) -> (show k ++ "," ++ show entities, pred)) v) (Map.toList posRelations)
  -- writeCsv (dataSetDir </> "new_pos_relations" ++ show indexNum ++ ".csv") newPosRelationsList
  -- let newNegRelationsList = concatMap (\(k, v) -> map (\(entities, pred) -> (show k ++ "," ++ show entities, pred)) v) (Map.toList negRelations)
  -- writeCsv (dataSetDir </> "new_neg_relations" ++ show indexNum ++ ".csv") newNegRelationsList

  let posRelations' = Map.map (map (\(xs, y) -> ((xs, y), 1.0))) posRelations
  let negRelations' = Map.map (map (\(xs, y) -> ((xs, y), 0.0))) negRelations
  let allRelations = Map.unionWith (++) posRelations' negRelations'

  let allRelationsList = concatMap (\(k, v) -> map (\x -> (k, x)) v) (Map.toList allRelations) :: [(Int, (([Int], Int), Float))]
  
  -- シャッフルを実行
  gen <- newStdGen
  let shuffledList = shuffle' allRelationsList (length allRelationsList) gen

  -- データを分割
  let (trainData, testData) = splitAt (round $ 0.70 * fromIntegral (length shuffledList)) shuffledList
  let testData = trainData
  putStrLn $ "Train Relations Count: " ++ show (length trainData)
  putStrLn $ "Test Relations Count: " ++ show (length testData)
  S.hFlush S.stdout

  -- 分割したデータをMapに戻す
  let trainRelations = Map.fromListWith (++) [(k, [v]) | (k, v) <- trainData]
  let testRelations = Map.fromListWith (++) [(k, [v]) | (k, v) <- testData]
  
  let modelName = "mlp-model" ++ show indexNum
  mapM_ (\(arity, relations) -> trainModel modelName relations arity) (Map.toList trainRelations)
  mapM_ (\(arity, relations) -> testModel modelName relations arity) (Map.toList testRelations)

-- CSVファイルを読み込む関数
readCsv :: FilePath -> IO [T.Text]
readCsv path = do
  content <- S.readFile path
  return $ T.lines (T.pack content)

-- CSVファイルに書き込む関数
writeCsv :: FilePath -> [(String, Int)] -> IO ()
writeCsv path content = do
  let textContent = map (\(k, v) -> T.unpack (T.pack (k ++ "," ++ show v))) content
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

  -- トレーニングとテストを実行
  processAndTrain ps posStr
  -- processAndTest ps testStr

testNeuralDTS :: IO()
testNeuralDTS = do
  -- CSVファイルを読み込む
  posStr <- readCsv (inputsDir ++ "/test2.csv")

  lr <- L.lexicalResourceBuilder Juman.KWJA
  let ps = CP.ParseSetting jpOptions lr 1 1 1 1 True Nothing Nothing True False

  -- トレーニングとテストを実行
  checkAccuracy ps posStr