{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module DTS.NeuralDTS.PreProcess (
  extractPredicateName,
--   strToEntityPred,
  getTrainRelations,
  getTestRelations,
  testPreterm
  ) where

import qualified Data.Text.Lazy as T      --text
import qualified Data.Text.Lazy.IO as T   --text
import qualified Data.List as L           --base
import qualified Data.Map as Map
import qualified Data.Maybe (mapMaybe)
import qualified Parser.ChartParser as CP
import qualified Parser.Language.Japanese.Templates as TP
import qualified Interface.HTML as HTML
import qualified Interface.Text as T
import qualified DTS.UDTT as UD
import qualified DTS.Prover.TypeChecker as Ty
import qualified DTS.Prover.Judgement as Ty

import qualified System.IO as S
import System.FilePath ((</>))
import Debug.Trace
import DTS.DTT (Preterm)
import GHC.IO (unsafePerformIO)
import qualified Data.Set as Set
import Data.Maybe (maybeToList)
import DTS.Prover (choice)
import Data.List (nub)
import Data.Map (mapMaybe)

dataDir = "src/DTS/NeuralDTS/DataSet"

-- 辞書をCSVに書き出す
writeCsv :: FilePath -> [(String, Int)] -> IO ()
writeCsv path dict = do
  let content = unlines $ map (\(name, idx) -> L.intercalate "," [name, show idx]) dict
  writeFile path content

-- CSVから辞書を読み込む
readCsv :: FilePath -> IO [(String, Int)]
readCsv path = do
  content <- readFile path
  return $ map (\line -> let [name, idx] = map T.unpack (T.splitOn "," (T.pack line)) in (name, read idx)) (lines content)

getTrainRelations :: Int -> Int -> [T.Text] -> IO [((Int, Int), Int)]
getTrainRelations beam nbest str = do
  let (entitiesIndex, predsIndex, groupedPreds) = strToEntityPred beam nbest str
      entityDict = map (\(ent, idx) -> (show ent, idx)) entitiesIndex
      predDict = map (\(pred, idx) -> (show pred, idx)) predsIndex
      
  -- 辞書をCSVに書き出し
  writeCsv (dataDir </> "entity_dict.csv") entityDict
  writeCsv (dataDir </> "predicate_dict.csv") predDict

  let binaryPreds = Map.findWithDefault [] 2 groupedPreds -- :: [UD.Preterm]
      relations = [((entity1ID, entity2ID), predID) |
                  pred <- binaryPreds, -- 2項述語を順に処理
                  predID <- maybeToList (lookupInListPartial (\key -> isPrefixOfPreterm (extractPredicateName pred) key) predsIndex),
                  (arg1, arg2) <- extractArguments pred,
                  entity1ID <- maybeToList (lookupInList arg1 entitiesIndex),
                  entity2ID <- maybeToList (lookupInList arg2 entitiesIndex)]

  putStrLn "Entity Dictionary written to entity_dict.csv"
  putStrLn "Predicate Dictionary written to predicate_dict.csv"
  return relations

getTestRelations :: Int -> Int -> [T.Text] -> IO [((Int, Int), Int)]
getTestRelations beam nbest str = do
  -- CSVから辞書を読み込み
  entityDict <- readCsv (dataDir </> "entity_dict.csv")
  predDict <- readCsv (dataDir </> "predicate_dict.csv")

  let (entitiesIndex, predsIndex, groupedPreds) = strToEntityPred beam nbest str
      binaryPreds = Map.findWithDefault [] 2 groupedPreds -- :: [UD.Preterm]
      relations = [((entity1ID, entity2ID), predID) |
                  pred <- binaryPreds, -- 2項述語を順に処理
                  predID <- maybeToList (lookupInListPartial (\key -> isPrefixOfPreterm (extractPredicateName pred) key) predsIndex),
                  (arg1, arg2) <- extractArguments pred,
                  entity1ID <- maybeToList (lookupInList arg1 entitiesIndex),
                  entity2ID <- maybeToList (lookupInList arg2 entitiesIndex)]
  putStrLn $ "Test Relations 2:" ++ show relations
  return relations

strToEntityPred :: Int -> Int -> [T.Text] -> ([(UD.Preterm, Int)], [(UD.Preterm, Int)], Map.Map Int [UD.Preterm])
strToEntityPred beam nbest str = unsafePerformIO $ do
  -- S1, S2, ... と文に番号を振る
  let numberedStr = zipWith (\i s -> (T.pack $ "S" ++ show i, s)) [1..] str
  nodeslist <- mapM (\(num, s) -> fmap (map (\n -> (num, n))) $ CP.simpleParse beam s) numberedStr
  
  let pairslist = map (map (\(num, node) -> (num, node, UD.betaReduce $ UD.sigmaElimination $ CP.sem node)) . take nbest) nodeslist
      chosenlist = choice pairslist
      nodeSRlist = map unzip3 chosenlist
      nds = concat $ map (\(_, nodes, _) -> nodes) nodeSRlist
      srs = concat $ map (\(nums, _, srs) -> zip nums srs) nodeSRlist -- :: [(T.Text, UD.Preterm)]
      sig = foldl L.union [] $ map CP.sig nds -- :: [(T.Text,Preterm)] (= UD.Signature)

  putStrLn $ "~~srs~~"
  printList srs 

  putStrLn $ "~~sig~~"
  printList sig

  let judges = Ty.getJudgements [] [((UD.Con x), y) | (x, _) <- srs, (_, y) <- srs] -- :: [([UJudgement], [UJudgement])]
      entitiesJudges = map fst judges -- :: [[UJudgement]]
      predsJudges = map snd judges -- :: [[UJudgement]]
      entities = map extractTermPreterm entitiesJudges -- :: [[Preterm]]
      correctPreds = map extractTypePreterm predsJudges -- :: [[Preterm]]

  let (tmp1, others) = L.partition isEntity [((UD.Con x), y) | (x, y) <- sig]
      allEntities = entities ++ [map fst tmp1]
      sigPreds = map fst others
  
  -- entitiesの辞書
  let entitiesIndex = pretermsIndex allEntities :: [(UD.Preterm, Int)]
  -- entityの総数
  let entitiesNum = length entitiesIndex
  putStrLn $ "~~entityの辞書~~ "
  putStrLn $ (show entitiesIndex) ++ " " ++ (show entitiesNum) ++ "個"
  -- id->entityのマップ
  -- let entityMap = Map.fromList entitiesIndex
  -- putStrLn $ "Entity Map: " ++ show entityMap

  -- predsの辞書
  let predsIndex = pretermsIndex [sigPreds]
  -- predsの総数
  let predsNum = length predsIndex
  putStrLn $ "~~述語の辞書~~ "
  putStrLn $ (show predsIndex) ++ " " ++ (show predsNum) ++ "個"
  -- id->述語のマップ
  -- let predsIdxMap = Maps.fromList predsIndex
  -- putStrLn $ "Predicate Map: " ++ show predsIdxMap
  -- putStrLn $ show entitiesNum ++ "," ++ show predsNum

   -- 成り立つpreds
  putStrLn $ "~~成り立つ述語~~ "
  let groupedPreds = groupPredicatesByArity $ concat correctPreds :: Map.Map Int [UD.Preterm]
  -- mapM_ (\(arity, preds) -> do
  --       putStrLn $ show arity ++ "項述語:"
  --       mapM_ (putStrLn . show) preds
  --       putStrLn ""
  --   ) $ Map.toList groupedPreds
  putStrLn $ show $ Map.toList groupedPreds

  -- id->述語のマップ
  -- let predsIdxMap = Map.fromList indexPreds
  -- putStrLn $ "Predicate Map: " ++ show predsIdxMap
  -- putStrLn $ show entitiesNum ++ "," ++ show predsNum

  return (entitiesIndex, predsIndex, groupedPreds)

lookupInList :: Eq a => a -> [(a, b)] -> Maybe b
lookupInList key list =
  case filter (\(k, _) -> k == key) list of
    ((_, value) : _) -> Just value
    []               -> Nothing

lookupInListPartial :: (Eq a, Show a, Show b) => (a -> Bool) -> [(a, b)] -> Maybe b
lookupInListPartial predicate list =
  let filtered = filter (\(k, _) -> predicate k) list
  in case filtered of
    ((_, value) : _) -> 
      Just value
      -- trace ("lookupInListPartial: Match found -> " ++ show key ++ " -> " ++ show value) $ Just value
    [] -> 
      Nothing
      -- trace ("lookupInListPartial: No match found") Nothing

isPrefixOfPreterm :: UD.Preterm -> UD.Preterm -> Bool
isPrefixOfPreterm (UD.Con key) (UD.Con pred) =
  let result = T.isPrefixOf key pred
  -- in trace ("isPrefixOfPreterm: Comparing " ++ show key ++ " with " ++ show pred ++ " -> " ++ show result) 
  in result
isPrefixOfPreterm _ _ =
  False
  -- trace "isPrefixOfPreterm: Types do not match for comparison" False

extractArguments :: UD.Preterm -> [(UD.Preterm, UD.Preterm)]
extractArguments (UD.App (UD.App _ arg1) arg2) = [(arg1, arg2)] -- 2項述語の場合
extractArguments _ = []

extractPredicateName :: UD.Preterm -> UD.Preterm
extractPredicateName (UD.Con name) =
  let simplified = T.takeWhile (/= '(') name
  -- in trace ("extractPredicateName: Simplifying " ++ show name ++ " -> " ++ show simplified) $ UD.Con simplified
  in UD.Con simplified
extractPredicateName (UD.App f arg) =
  -- trace ("extractPredicateName: Encountered App structure: " ++ show f ++ " applied to " ++ show arg) $
  extractPredicateName f
extractPredicateName preterm =
  -- trace ("extractPredicateName: Non-Con type encountered: " ++ show preterm) preterm
  preterm

testPreterm :: IO()
testPreterm = do
  let predsIndex = [(UD.Con "走る/はしる/ガヲニ", 0), (UD.Con "歌/か;歌/うた", 1)]
      binaryPreds = [UD.Con "歌/か;歌/うた(π1(S1),次郎/じろう)", UD.Con "存在しない述語"]

  let simplifiedBinaryPreds = map extractPredicateName binaryPreds
  putStrLn "Debug: Simplified binaryPreds"
  putStrLn $ show simplifiedBinaryPreds

  mapM_ (\pred -> print $ lookupInListPartial (\key -> isPrefixOfPreterm pred key) predsIndex) simplifiedBinaryPreds

indexPreterms :: [[UD.Preterm]] -> [(Int, UD.Preterm)]
indexPreterms = snd . L.foldl' addIndexedGroup (0, [])
  where
    addIndexedGroup :: (Int, [(Int, UD.Preterm)]) -> [UD.Preterm] -> (Int, [(Int, UD.Preterm)])
    addIndexedGroup (startIndex, acc) group = 
      let indexed = zip [startIndex..] group
          newIndex = startIndex + length group
      in (newIndex, acc ++ indexed)

pretermsIndex :: [[UD.Preterm]] -> [(UD.Preterm, Int)]
pretermsIndex = snd . L.foldl' addIndexedGroup (0, [])
  where
    addIndexedGroup :: (Int, [(UD.Preterm, Int)]) -> [UD.Preterm] -> (Int, [(UD.Preterm, Int)])
    addIndexedGroup (startIndex, acc) group = 
      let indexed = [(term, index) | (index, term) <- zip [startIndex..] group]
          newIndex = startIndex + length group
      in (newIndex, acc ++ indexed)

extractTermPreterm :: [Ty.UJudgement] -> [UD.Preterm]
extractTermPreterm = map (\(Ty.UJudgement _ preterm _) -> preterm)
extractTypePreterm :: [Ty.UJudgement] -> [UD.Preterm]
extractTypePreterm = map (\(Ty.UJudgement _ _ preterm) -> preterm)

printList :: [(T.Text, UD.Preterm)] -> IO ()
printList [] = return ()
printList ((text, preterm):xs) = do
    T.putStr "Text: "
    T.putStrLn text
    putStr "Preterm: "
    print preterm
    printList xs
    putStr ""

isEntity :: (UD.Preterm, UD.Preterm) -> Bool
isEntity (_, (UD.Con cname)) = cname == "entity"
isEntity _ = False

-- isPred :: (UD.Preterm, UD.Preterm) -> Bool
-- isPred (tm, ty) = 
--   trace ("tm : " ++ show tm ++ "ty : " ++ show ty) $
--   case ty of
--     UD.App f x -> True
--     _ -> False

containsFunctionType :: UD.Preterm -> Bool
containsFunctionType term = case term of
    UD.Pi _ _ -> True
    UD.Lam _ -> True
    UD.App f x -> containsFunctionType f || containsFunctionType x
    _ -> False

groupPredicatesByArity :: [UD.Preterm] -> Map.Map Int [UD.Preterm]
groupPredicatesByArity predicates = 
    Map.fromListWith (++) $ groupSingle predicates
  where
    groupSingle preds = [(countArgs p, [p]) | p <- preds]

countArgs :: UD.Preterm -> Int
-- countArgs term = trace ("Counting args for: " ++ show term) $ countArgsFromString (show term)
countArgs term = countArgsFromString (show term)

countArgsFromString :: String -> Int
countArgsFromString s = 
    let withoutOuterParens = T.pack $ init $ tail $ dropWhile (/= '(') s
        args = T.splitOn (T.pack ",") withoutOuterParens
    -- in trace ("Split args: " ++ show args) $
    in length args
