{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module DTS.NeuralDTS.PreProcess (
  extractPredicateName,
--   strToEntityPred,
  getTrainRelations,
  getTestRelations
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
import Control.Monad (unless)
import Data.Maybe (maybeToList, fromJust)
import DTS.Prover (choice)
import Data.List (nub)
import Data.Map (mapMaybe)
import System.Directory (doesFileExist)
import System.Random (randomRIO)

dataDir = "src/DTS/NeuralDTS/dataSet"

writeCsv :: FilePath -> [(String, Int)] -> IO ()
writeCsv path dict = S.withFile path S.WriteMode $ \h -> do
  let content = unlines $ map (\(name, idx) -> L.intercalate "," [name, show idx]) dict
  S.hPutStr h content

-- CSVから辞書を読み込む
readCsv :: FilePath -> IO [(String, Int)]
readCsv path = S.withFile path S.ReadMode $ \h -> do
  content <- S.hGetContents h
  let linesContent = lines content
      result = map (\line -> let [name, idx] = map T.unpack (T.splitOn "," (T.pack line)) in (name, read idx)) linesContent
  length result `seq` return result

ensureFileExists :: FilePath -> IO ()
ensureFileExists path = do
  exists <- doesFileExist path
  unless exists $ writeFile path ""

writeRelationsCsv :: FilePath -> [([Int], Int)] -> IO ()
writeRelationsCsv path relations = S.withFile path S.WriteMode $ \h -> do
  let content = unlines $ map (\(entities, p) -> L.intercalate "," (map show entities ++ [show p])) relations
  S.hPutStr h content

-- n項述語に対応するための関数
getTrainRelations :: Int -> Int -> [T.Text] -> IO (Map.Map Int [([Int], Int)], Map.Map Int [([Int], Int)])
getTrainRelations beam nbest posStr = do
  let posStrIndexed = zipWith (\i s -> (T.pack $ "S" ++ show i, s)) [1..] posStr

  -- 正解データのgroupedPredsを作成
  let (posEntities, posPreds, posGroupedPreds) = strToEntityPred beam nbest posStrIndexed

  -- エンティティの重複を取り除く
  let uniqueEntities = L.nub posEntities
      entitiesIndex = zip (map show uniqueEntities) [0..] :: [(String, Int)]
      predsIndex = zip (map show posPreds) [0..] :: [(String, Int)]
      entitiesMap = Map.fromList entitiesIndex :: Map.Map String Int
      predsMap = Map.fromList predsIndex :: Map.Map String Int

  -- putStrLn "~~entitiesMap~~"
  -- print entitiesMap
  -- putStrLn "~~predsMap~~"
  -- print predsMap
  -- putStrLn "~~posGroupedPreds~~"
  -- print posGroupedPreds

  -- 辞書をCSVに書き出し
  writeCsv (dataDir </> "entity_dict.csv") entitiesIndex
  writeCsv (dataDir </> "predicate_dict.csv") predsIndex

  -- let binaryPosPreds = Map.findWithDefault [] 2 posGroupedPreds -- :: [UD.Preterm]
  --     posRelations = [((entity1ID, entity2ID), predID) |
  --                     pred <- binaryPosPreds, -- 2項述語を順に処理
  --                     predID <- maybeToList (Map.lookup (show (extractPredicateName pred)) predsMap),
  --                     (arg1, arg2) <- extractArguments pred,
  --                     entity1ID <- maybeToList (Map.lookup (show arg1) entitiesMap),
  --                     entity2ID <- maybeToList (Map.lookup (show arg2) entitiesMap)]

  -- n項述語ごとに成り立つ述語を分ける
  let posRelationsByArity = Map.mapWithKey (\arity preds -> 
        [ (entityIDs, predID) |
          pred <- preds, -- n項述語を順に処理
          predID <- maybeToList (Map.lookup (show (extractPredicateName pred)) predsMap),
          let args = extractArguments pred,
          let entityIDs = concatMap (\arg -> maybeToList (Map.lookup (show arg) entitiesMap)) args
        ]
        ) posGroupedPreds :: (Map.Map Int [([Int], Int)])

  -- putStrLn "~~posRelationsByArity~~"
  -- print posRelationsByArity

  -- ネガティブデータを作成
  negRelationsByArityList <- mapM (\(arity, posRelations) -> do
    let posRelationSet = Set.fromList posRelations
        allEntityCombinations = sequence $ replicate arity (Map.keys entitiesMap)
        allPreds = Map.keys predsMap
    negRelations <- generateNegRelations posRelationSet allEntityCombinations allPreds entitiesMap predsMap (length posRelations)
    return (arity, negRelations)
    ) (Map.toList posRelationsByArity)
  let negRelationsByArity = Map.fromList negRelationsByArityList

  -- 成り立つ述語のファイルをn項ごとに分けて保存
  mapM_ (\(arity, posRelations) -> writeRelationsCsv (dataDir </> "pos_relations_arity" ++ show arity ++ ".csv") posRelations) (Map.toList posRelationsByArity)
  mapM_ (\(arity, negRelations) -> writeRelationsCsv (dataDir </> "neg_relations_arity" ++ show arity ++ ".csv") negRelations) (Map.toList negRelationsByArity)

  putStrLn "Entity Dictionary written to entity_dict.csv"
  putStrLn "Predicate Dictionary written to predicate_dict.csv"
  return (posRelationsByArity, negRelationsByArity)

getTestRelations :: Int -> Int -> [T.Text] -> IO (Map.Map Int [([Int], Int)])
getTestRelations beam nbest str = do
  -- CSVから辞書を読み込み
  entityDictList <- readCsv (dataDir </> "entity_dict.csv")
  predDictList <- readCsv (dataDir </> "predicate_dict.csv")
  let entityDict = Map.fromList entityDictList
      predDict = Map.fromList predDictList :: Map.Map String Int

  let strIndexed = zipWith (\i s -> (T.pack $ "TestS" ++ show i, s)) [1..] str
      (entities, preds, groupedPreds) = strToEntityPred beam nbest strIndexed

  let entitiesIndexDict = zipWith (\ent idx -> (T.pack (show ent), idx)) entities [0..] :: [(T.Text, Int)]
      predsIndexDict = zipWith (\pred idx -> (T.pack (show pred), idx)) preds [0..] :: [(T.Text, Int)]
  let updatedEntityDict = foldl (\dict (ent, _) -> if Map.member (T.unpack ent) dict then dict else Map.insert (T.unpack ent) (Map.size dict) dict) entityDict entitiesIndexDict
      updatedPredDict = foldl (\dict (pred, _) -> if Map.member (T.unpack pred) dict then dict else Map.insert (T.unpack pred) (Map.size dict) dict) predDict predsIndexDict
  writeCsv (dataDir </> "entity_dict.csv") (Map.toList updatedEntityDict)
  writeCsv (dataDir </> "predicate_dict.csv") (Map.toList updatedPredDict)

  let testRelationsByArity = Map.mapWithKey (\arity preds ->
        [ (map (\arg -> fromJust $ Map.lookup (show arg) updatedEntityDict) args, predID)
        | pred <- preds,
          predID <- maybeToList (Map.lookup (show (extractPredicateName pred)) updatedPredDict),
          let args = extractArguments pred,
          all (\arg -> Map.member (show arg) updatedEntityDict) args
        ]
        ) groupedPreds :: Map.Map Int [([Int], Int)]

  putStrLn "~~testRelations~~"
  print testRelationsByArity
  return testRelationsByArity

-- strToEntityPred :: Int -> Int -> [T.Text] -> ([(UD.Preterm, Int)], [(UD.Preterm, Int)], Map.Map Int [UD.Preterm])
strToEntityPred :: Int -> Int -> [(T.Text, T.Text)] -> ([UD.Preterm], [UD.Preterm], Map.Map Int [UD.Preterm])
strToEntityPred beam nbest numberedStr = unsafePerformIO $ do
  -- S1, S2, ... と文に番号を振る
  -- let numberedStr = zipWith (\i s -> (T.pack $ "S" ++ show i, s)) [1..] str :: [(T.Text, T.Text)]
  -- putStrLn "~~numberedStr~~"
  -- putStrLn $ show numberedStr
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

  let judges = getJudgements [] [((UD.Con x), y) | (x, _) <- srs, (_, y) <- srs] -- :: [([UJudgement], [UJudgement])]
      entitiesJudges = map fst judges -- :: [[UJudgement]]
      predsJudges = map snd judges -- :: [[UJudgement]]
      entities = map extractTermPreterm entitiesJudges -- :: [[Preterm]]
      correctPreds = map extractTypePreterm predsJudges -- :: [[Preterm]]

  let (tmp1, others) = L.partition isEntity [((UD.Con x), y) | (x, y) <- sig]
      allEntities = concat entities ++ map fst tmp1
      sigPreds = map fst others

   -- 成り立つpreds
  putStrLn "~~成り立つ述語~~ "
  let groupedPreds = groupPredicatesByArity $ concat correctPreds :: Map.Map Int [UD.Preterm]
  print (Map.toList groupedPreds)

  return (allEntities, sigPreds, groupedPreds)

extractArguments :: UD.Preterm -> [UD.Preterm]
extractArguments (UD.App f arg) = extractArguments f ++ [arg]
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

generateNegRelations :: Set.Set ([Int], Int) -> [[String]] -> [String] -> Map.Map String Int -> Map.Map String Int -> Int -> IO [([Int], Int)]
generateNegRelations posRelationSet allEntityCombinations allPreds entitiesMap predsMap numNegRelations = do
  negRelations <- generateNegRelations' posRelationSet allEntityCombinations allPreds entitiesMap predsMap numNegRelations []
  return (take numNegRelations negRelations)

generateNegRelations' :: Set.Set ([Int], Int) -> [[String]] -> [String] -> Map.Map String Int -> Map.Map String Int -> Int -> [([Int], Int)] -> IO [([Int], Int)]
generateNegRelations' posRelationSet allEntityCombinations allPreds entitiesMap predsMap numNegRelations negRelations
  | length negRelations >= numNegRelations = return negRelations
  | otherwise = do
      entityCombination <- randomChoice allEntityCombinations
      pred <- randomChoice allPreds
      let entityIDs = map (entitiesMap Map.!) entityCombination
          predID = predsMap Map.! pred
          negRelation = (entityIDs, predID)
      if Set.member negRelation posRelationSet || elem negRelation negRelations
        then generateNegRelations' posRelationSet allEntityCombinations allPreds entitiesMap predsMap numNegRelations negRelations
        else generateNegRelations' posRelationSet allEntityCombinations allPreds entitiesMap predsMap numNegRelations (negRelation : negRelations)

randomChoice :: [a] -> IO a
randomChoice xs = do
  idx <- randomRIO (0, length xs - 1)
  return (xs !! idx)

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

isPred :: (UD.Preterm, UD.Preterm) -> Bool
isPred (tm, ty) = 
  trace ("tm : " ++ show tm ++ "ty : " ++ show ty) $
  case ty of
    UD.App f x -> True
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

-- termとtypeを受け取って([entity], [述語])のlistを得る
getJudgements :: Ty.TUEnv -> [(UD.Preterm, UD.Preterm)] -> [([Ty.UJudgement], [Ty.UJudgement])]
getJudgements env [] = []
getJudgements env ((tm, ty):rest) =
  let newPairs = loop env (tm, ty)
      newJudgements = map (\(tm2, ty2) -> Ty.UJudgement env tm2 ty2) newPairs
      (entities, others) = L.partition isEntity newJudgements
      (preds, _) = L.partition isPred others
  in  ((entities, preds) : getJudgements env rest)
  where
      -- isEntity (UJudgement _ _ (UD.Con cname)) = cname == "entity"
      isEntity (Ty.UJudgement _ _ (UD.Con _)) = True
      isEntity _ = False
      isPred (Ty.UJudgement _ _ term) = 
          case term of
            UD.App f x ->
                not (containsFunctionType f) && 
                not (containsFunctionType x)
            _ -> False

containsFunctionType :: UD.Preterm -> Bool
containsFunctionType term = case term of
    UD.Pi _ _ -> True
    UD.Lam _ -> True
    UD.App f x -> containsFunctionType f || containsFunctionType x
    _ -> False

loop :: Ty.TUEnv -> (UD.Preterm, UD.Preterm) -> [(UD.Preterm, UD.Preterm)]
loop env (tm, ty) = case ty of
    UD.Sigma _ _ ->
      let sigmaResult = sigmaE (tm, ty)
      in concatMap (loop env) sigmaResult
    _ -> [(tm, ty)]

sigmaE :: (UD.Preterm, UD.Preterm) -> [(UD.Preterm, UD.Preterm)]
sigmaE (m, (UD.Sigma a b)) = [((UD.Proj UD.Fst m), a), ((UD.Proj UD.Snd m), (UD.subst b (UD.Proj UD.Fst m) 0))]
