{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module DTS.NeuralDTS.PreProcess (
  extractPredicateName
  ,getTrainRelations
  -- getTestRelations
  ) where

import Control.Monad (unless, replicateM, (>=>))
import Control.Monad.RWS (MonadState(put))
import Control.Concurrent.Async (mapConcurrently)
import Control.Exception (try, SomeException)
import qualified Data.Text.Lazy as T      --text
import qualified ListT
import qualified Data.Text.Lazy.IO as T   --text
import qualified Data.List as L           --base
import qualified Data.Map as Map
import qualified Data.Maybe (fromJust, mapMaybe)
import qualified Data.Set as Set
import Data.List.Split (chunksOf)
import Debug.Trace
import GHC.IO (unsafePerformIO)
import qualified System.IO as S
import System.FilePath ((</>))
import System.Directory (doesFileExist)
import System.Random (randomRIO, newStdGen, randomRs)
import System.Random.Shuffle (shuffle)
import Database.SQLite.Simple (Connection)

import qualified Parser.CCG as CCG
import qualified Parser.ChartParser as CP
import qualified Parser.PartialParsing as Partial
import qualified Parser.Language.Japanese.Templates as TP
import qualified Interface as I
import qualified Interface.HTML as HTML
import qualified Interface.Text as T
import qualified DTS.DTTdeBruijn as DTT
import qualified DTS.UDTTdeBruijn as UDTT
import qualified DTS.QueryTypes as QT
import qualified DTS.NaturalLanguageInference as NLI
import qualified DTS.NeuralDTS.WordNet.WordNet as WN

dataDir = "src/DTS/NeuralDTS/dataSet"
indexNum = 11

writeCsv :: FilePath -> [(String, Int)] -> IO ()
writeCsv path dict = S.withFile path S.WriteMode $ \h -> do
  let content = unlines $ map (\(name, idx) -> L.intercalate "," [name, show idx]) dict
  S.hPutStr h content
  S.hFlush S.stdout

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
getTrainRelations :: CP.ParseSetting -> [T.Text] ->
  IO ((Map.Map Int [([Int], Int)], Map.Map Int [([Int], Int)]), (Map.Map Int [([Int], Int)], Map.Map Int [([Int], Int)]))
getTrainRelations ps posStr = do
  putStrLn $ "~~getTrainRelations~~"
  let posStrIndexed = zipWith (\i s -> (T.pack $ "S" ++ show i, s)) [1..] posStr
  let (posEntities, posPreds, (posOrgData, posAddData)) = strToEntityPred ps posStrIndexed

  -- posGroupedPredsに含まれているエンティティと述語をフィルタリング
  let includedPreds1 = L.nub $ concatMap (map fst) (Map.elems posOrgData)
      includedPreds2 = L.nub $ concatMap (map fst) (Map.elems posAddData)
      includedEntities1 = L.nub $ concatMap (map snd) (Map.elems posOrgData)
      includedEntities2 = L.nub $ concatMap (map snd) (Map.elems posAddData)
      includedPreds = includedPreds1 ++ includedPreds2
      includedEntities = includedEntities1 ++ includedEntities2

  -- posEntitiesとposPredsをフィルタリング
  let filteredEntities = filter (`elem` concat includedEntities) posEntities
      filteredPreds = filter (`elem` includedPreds) posPreds

  -- -- フィルタリング前後の要素数を出力
  putStrLn $ "Entities reduced: " ++ show (length posEntities) ++ "->" ++ show (length filteredEntities)
  putStrLn $ "Preds reduced: " ++ show (length posPreds) ++ "->" ++ show (length filteredPreds)

  -- エンティティの重複を取り除く
  let uniqueEntities = L.nub filteredEntities
      entitiesIndex = zip (map show uniqueEntities) [0..] :: [(String, Int)]
      predsIndex = zip (map show filteredPreds) [0..] :: [(String, Int)]
      entitiesMap = Map.fromList entitiesIndex :: Map.Map String Int
      predsMap = Map.fromList predsIndex :: Map.Map String Int

  -- 辞書をCSVに書き出し
  writeCsv (dataDir </> "entity_dict" ++ show indexNum ++ ".csv") entitiesIndex
  writeCsv (dataDir </> "predicate_dict" ++ show indexNum ++ ".csv") predsIndex

  putStrLn $ "Entity Dictionary written to entity_dict" ++ show indexNum ++ ".csv"
  putStrLn $ "Predicate Dictionary written to predicate_dict" ++ show indexNum ++ ".csv"
  S.hFlush S.stdout

  let posOrgData' = Map.filterWithKey (\arity _ -> arity == 2) posOrgData
  let posAddData' = Map.filterWithKey (\arity _ -> arity == 2) posAddData

  let posOrgRelationsByArity = Map.mapWithKey (\arity preds ->
        Data.Maybe.mapMaybe (\(pred, args) -> do
          let predName = show (extractPredicateName pred)
          predID <- Map.lookup predName predsMap
          let entityIDs = Data.Maybe.mapMaybe (\arg ->
                let argName = show arg
                in Map.lookup argName entitiesMap
                ) args
          return (entityIDs, predID)
        ) preds
        ) posOrgData'
  let posAddRelationsByArity = Map.mapWithKey (\arity preds ->
        Data.Maybe.mapMaybe (\(pred, args) -> do
          let predName = show (extractPredicateName pred)
          predID <- Map.lookup predName predsMap
          let entityIDs = Data.Maybe.mapMaybe (\arg ->
                let argName = show arg
                in Map.lookup argName entitiesMap
                ) args
          return (entityIDs, predID)
        ) preds
        ) posAddData'
  -- putStrLn $ "~~posRelationsByArity~~"
  -- print posRelationsByArity
  mapM_ (\(arity, posRelations) -> writeRelationsCsv (dataDir </> "pos_org_relations_arity" ++ show arity ++ "_" ++ show indexNum ++ ".csv")
    posRelations) (Map.toList posOrgRelationsByArity)
  mapM_ (\(arity, posRelations) -> writeRelationsCsv (dataDir </> "pos_add_relations_arity" ++ show arity ++ "_" ++ show indexNum ++ ".csv")
    posRelations) (Map.toList posAddRelationsByArity)
  putStrLn $ "posRelation written to pos_relations_arity" ++ "_" ++ show indexNum ++ ".csv"
  S.hFlush S.stdout

  -- ネガティブデータを作成
  negOrgRelationsByArityList <- mapM (\(arity, posRelations) -> do
    let posRelationSet = Set.fromList posRelations
        allPreds = Map.elems predsMap
        numEntities = Map.size entitiesMap
    negRelations <- generateNegRelations posRelationSet allPreds numEntities arity (length posRelations)
    return (arity, negRelations)
    ) (Map.toList posOrgRelationsByArity)
  let negOrgRelationsByArity = Map.fromList negOrgRelationsByArityList
  negAddRelationsByArityList <- mapM (\(arity, posRelations) -> do
    let posRelationSet = Set.fromList posRelations
        allPreds = Map.elems predsMap
        numEntities = Map.size entitiesMap
    negRelations <- generateNegRelations posRelationSet allPreds numEntities arity (length posRelations)
    return (arity, negRelations)
    ) (Map.toList posAddRelationsByArity)
  let negAddRelationsByArity = Map.fromList negAddRelationsByArityList
  mapM_ (\(arity, negRelations) -> writeRelationsCsv (dataDir </> "neg_org_relations_arity" ++ show arity ++ "_" ++ show indexNum ++ ".csv")
    negRelations) (Map.toList negOrgRelationsByArity)
  mapM_ (\(arity, negRelations) -> writeRelationsCsv (dataDir </> "neg_add_relations_arity" ++ show arity ++ "_" ++ show indexNum ++ ".csv")
    negRelations) (Map.toList negAddRelationsByArity)

  putStrLn $ "negRelation written to neg_relations_arity" ++ "_" ++ show indexNum ++ ".csv"
  S.hFlush S.stdout
  return ((posOrgRelationsByArity, posAddRelationsByArity), (negOrgRelationsByArity, negAddRelationsByArity))

processBatch :: CP.ParseSetting -> [(T.Text, T.Text)] -> IO [[(T.Text, CCG.Node)]]
processBatch ps batch = do
  results <- mapM (\(num, s) -> do
                      result <- try (Partial.simpleParse ps s) :: IO (Either SomeException [CCG.Node])
                      case result of
                        Left ex -> do
                          putStrLn $ "Error parsing sentence: " ++ T.unpack s
                          putStrLn $ "Exception: " ++ show ex
                          return []
                        Right nodes -> return $ map (\n -> (num, n)) nodes
                  ) batch
  return results

strToEntityPred :: CP.ParseSetting -> [(T.Text, T.Text)] ->
  ([DTT.Preterm], [DTT.Preterm], (Map.Map Int [(DTT.Preterm, [DTT.Preterm])], Map.Map Int [(DTT.Preterm, [DTT.Preterm])]))
strToEntityPred ps strIndexed = unsafePerformIO $ do
  putStrLn $ "~~strToEntityPred~~"
  S.hFlush S.stdout
  -- バッチ処理を実行
  let batchSize = 10
  let batches = chunksOf batchSize strIndexed

  nodeslist <- fmap concat $ mapConcurrently (processBatch ps) batches
  -- putStrLn $ "~~nodeslist~~"
  -- print nodeslist
  -- DTTに変換
  let convertedNodes = map (map (\(num, node) -> (num, node, UDTT.toDTT $ CCG.sem node))) nodeslist :: [[(T.Text, CCG.Node, Maybe DTT.Preterm)]]
  -- putStrLn $ "~~convertedNodes~~"
  -- print convertedNodes
  -- 変換が失敗した場合のエラーハンドリング
  let handleConversion (num, node, Nothing) = trace (show num ++ ": toDTT error") Nothing
      handleConversion (num, node, Just dtt) = Just (num, node, DTT.betaReduce $ DTT.sigmaElimination dtt)
  -- let pairslist = map (map (\(num, node) -> (num, node, DTT.betaReduce $ DTT.sigmaElimination $ CCG.sem node)) . take 1) nodeslist;

  let pairslist = map (Data.Maybe.mapMaybe handleConversion . take 1) convertedNodes :: [[(T.Text, CCG.Node, DTT.Preterm)]]
      nonEmptyPairsList = filter (not . null) pairslist
      chosenlist = choice nonEmptyPairsList
      nodeSRlist = map unzip3 chosenlist
      nds = concat $ map (\(_, nodes, _) -> nodes) nodeSRlist
      srs = concat $ map (\(nums, _, srs) -> zip nums srs) nodeSRlist -- :: [(T.Text, DTT.Preterm)]
      sig = foldl L.union [] $ map CP.sig nds

  let successCount = length $ filter (not . null) pairslist
  putStrLn $ "Number of successful toDTT conversions: " ++ show successCount
  S.hFlush S.stdout

  let judges = concat $ map (\(num, sr) -> getJudgements sig [] [(DTT.Con num, sr)]) srs -- :: [[([DTT.Judgment], [DTT.Judgment])]]

  -- putStrLn $ "~~judges~~"
  -- print judges
  let entitiesJudges = map fst judges -- :: [[UJudgement]]
      predsJudges = map snd judges -- :: [[UJudgement]]
      entities = map extractTermPreterm entitiesJudges -- :: [[Preterm]]
      correctPreds = map extractTypePreterm predsJudges -- :: [[Preterm]]

  let transformedPreds = map transformPreterm $ concat correctPreds :: [(DTT.Preterm, [DTT.Preterm])]

  let (sigEntities, sigPreds) = L.partition isEntity [((DTT.Con x), y) | (x, y) <- sig]

  -- putStrLn $ "~~SigEntities~~"
  -- print sigEntities
  -- putStrLn $ "~~SigPreds~~"
  -- print sigPreds

  conn <- WN.openDatabase
  let sigEntities' = map (\(DTT.Con txt, _) -> (txt, DTT.Con txt)) sigEntities :: [(T.Text, DTT.Preterm)]
      sigPreds' = map (\(DTT.Con txt, _) -> (txt, DTT.Con txt)) sigPreds :: [(T.Text, DTT.Preterm)]
  -- synonymMapEntities <- augmentWithSynonyms conn sigEntities' -- :: Map (Text, Preterm) [(Text, Preterm)]
  synonymMapPreds <- augmentWithSynonyms conn sigPreds' -- :: Map Text [(Text, Preterm)]
  WN.closeDatabase conn

  let allEntities = concat entities ++ map fst sigEntities :: [DTT.Preterm]
  let allPreds = concatMap (\(original, synonyms) -> (DTT.Con original) : map (DTT.Con . fst) synonyms) (Map.toList synonymMapPreds) :: [DTT.Preterm]

  putStrLn $ "pred数の変化"
  putStrLn $ show (length sigPreds') ++ " -> " ++ show (length allPreds)

  -- let synonymMapEntities' = Map.fromListWith (++) [(word, map fst synonyms) | ((word, _), synonyms) <- Map.toList synonymMapEntities] :: Map.Map T.Text [T.Text]
  let synonymMapPreds' = Map.fromListWith (++) [(word, map fst synonyms) | ((word), synonyms) <- Map.toList synonymMapPreds] :: Map.Map T.Text [T.Text]
  let (orgPreds, addPreds) = replacePredicates synonymMapPreds' transformedPreds
  
  let originalGroupedPreds = groupPredicatesByArity orgPreds
  let addedGroupedPreds = groupPredicatesByArity addPreds

  putStrLn $ "~~Original Grouped Predicates Count by Arity~~"
  mapM_ (\(arity, preds) -> putStrLn $ "Arity " ++ show arity ++ ": " ++ show (length preds)) (Map.toList originalGroupedPreds)
  putStrLn $ "~~Added Grouped Predicates Count by Arity~~"
  mapM_ print (Map.toList originalGroupedPreds)
  mapM_ (\(arity, preds) -> putStrLn $ "Arity " ++ show arity ++ ": " ++ show (length preds)) (Map.toList addedGroupedPreds)
  mapM_ print (Map.toList addedGroupedPreds)

  return (allEntities, allPreds, (originalGroupedPreds, addedGroupedPreds))

augmentWithSynonyms :: Connection -> [(T.Text, DTT.Preterm)] -> IO (Map.Map T.Text [(T.Text, DTT.Preterm)])
augmentWithSynonyms conn sigs = do
  synonymMap <- fmap Map.fromList $ mapM (\(word, preterm) -> do
    synonyms <- WN.getSynonyms conn word -- :: [T.Text]
    let augmented = take 5 $ map (\syn -> (syn, preterm)) synonyms
    return (word, augmented)) sigs
  return synonymMap

replacePredicates :: Map.Map T.Text [T.Text] -> [(DTT.Preterm, [DTT.Preterm])] -> ([(DTT.Preterm, [DTT.Preterm])], [(DTT.Preterm, [DTT.Preterm])])
replacePredicates synonymMap relations = foldr replacePredicate ([], []) relations
  where
    replacePredicate (pred, args) (originals, addeds) =
      let predText = case pred of
                        DTT.Con name -> name
                        _ -> error "Expected DTT.Con"
          replacements = case Map.lookup predText synonymMap of
                           Just synonyms -> map (\syn -> (DTT.Con syn, args)) synonyms
                           Nothing -> []
      in ((DTT.Con predText, args) : originals, replacements ++ addeds)

choice :: [[a]] -> [[a]]
choice [] = [[]]
choice (a:as) = [x:xs | x <- a, xs <- choice as]

extractPredicateName :: DTT.Preterm -> DTT.Preterm
extractPredicateName (DTT.Con name) =
  let simplified = T.takeWhile (/= '(') name
  -- in trace ("extractPredicateName: Simplifying " ++ show name ++ " -> " ++ show simplified) $ DTT.Con simplified
  in DTT.Con simplified
extractPredicateName (DTT.App f arg) =
  -- trace ("extractPredicateName: Encountered App structure: " ++ show f ++ " applied to " ++ show arg) $
  extractPredicateName f
extractPredicateName preterm =
  -- trace ("extractPredicateName: Non-Con type encountered: " ++ show preterm) preterm
  preterm

-- TODO shuffle関数を使う
generateNegRelations :: Set.Set ([Int], Int) -> [Int] -> Int -> Int -> Int -> IO [([Int], Int)]
generateNegRelations posRelationSet allPreds numEntities arity numNegRelations = do
  gen <- newStdGen
  let generateSingleNegRelation = do
        entityCombination <- replicateM arity (randomRIO (0, numEntities - 1))
        pred <- randomRIO (0, length allPreds - 1)
        let negRelation = (entityCombination, allPreds !! pred)
        if Set.member negRelation posRelationSet
          then generateSingleNegRelation
          else return negRelation
  mapM (const generateSingleNegRelation) [1..numNegRelations]

randomChoice :: [a] -> IO a
randomChoice xs = do
  idx <- randomRIO (0, length xs - 1)
  return (xs !! idx)

extractTermPreterm :: [DTT.Judgment] -> [DTT.Preterm]
extractTermPreterm = map (\(DTT.Judgment _ _ preterm _) -> preterm)
extractTypePreterm :: [DTT.Judgment] -> [DTT.Preterm]
extractTypePreterm = map (\(DTT.Judgment _ _ _ preterm) -> preterm)

isEntity :: (DTT.Preterm, DTT.Preterm) -> Bool
isEntity (tm, ty) =
  -- trace ("isEntity tm : " ++ show tm ++ " ty : " ++ show ty) $
  case show ty of
    "entity" -> True
    _ -> False

isPred :: (DTT.Preterm, DTT.Preterm) -> Bool
isPred (tm, ty) =
  -- trace ("isPred tm : " ++ show tm ++ " ty : " ++ show ty) $
  case ty of
    (DTT.App f x) -> True
    _ -> False

-- groupPredicatesByArity :: [(DTT.Preterm, [DTT.Preterm])] -> [(Int, (DTT.Preterm, [DTT.Preterm]))]
groupPredicatesByArity :: [(DTT.Preterm, [DTT.Preterm])] -> Map.Map Int [(DTT.Preterm, [DTT.Preterm])]
groupPredicatesByArity predicates =
  Map.fromListWith (++) $ groupSingle predicates
      where
          groupSingle preds = [(length args, [(p, args)]) | (p, args) <- preds]

transformPreterm :: DTT.Preterm -> (DTT.Preterm, [DTT.Preterm])
transformPreterm term = case term of
  DTT.App f x ->
    let (func, args) = collectArgs f [x]
    in (func, args)
  _ -> (term, [])

collectArgs :: DTT.Preterm -> [DTT.Preterm] -> (DTT.Preterm, [DTT.Preterm])
collectArgs (DTT.App f x) args = collectArgs f (x : args)
collectArgs func args = (func, args)

-- termとtypeを受け取って([entity], [述語])のlistを得る
getJudgements :: DTT.Signature -> DTT.Context -> [(DTT.Preterm, DTT.Preterm)] -> [([DTT.Judgment], [DTT.Judgment])]
getJudgements _ _ [] = []
getJudgements sig cxt ((tm, ty):rest) =
  let newPairs = loop sig cxt (tm, ty)
      newJudgements = map (\(tm2, ty2) -> (DTT.Judgment sig cxt tm2 ty2)) newPairs
      (entities, others) = L.partition isEntityForJudgement newJudgements
      (preds, _) = L.partition isPredForJudgement others
  in  ((entities, preds) : getJudgements sig cxt rest)
  where
      isEntityForJudgement (DTT.Judgment _ _ tm3 ty3) = isEntity (tm3, ty3)
      -- isPredForJudgement (DTT.Judgment _ _ tm3 ty3) = isPred (tm3, ty3)
      isPredForJudgement (DTT.Judgment _ _ _ ty3) =
          case ty3 of
            DTT.App f x ->
                not (containsFunctionType f) &&
                not (containsFunctionType x)
            _ -> False

containsFunctionType :: DTT.Preterm -> Bool
containsFunctionType term = case term of
    DTT.Pi _ _ -> True
    DTT.Lam _ -> True
    DTT.App f x -> containsFunctionType f || containsFunctionType x
    _ -> False

loop :: DTT.Signature -> DTT.Context -> (DTT.Preterm, DTT.Preterm) -> [(DTT.Preterm, DTT.Preterm)]
loop sig cxt (tm, ty) = case ty of
    DTT.Sigma _ _ ->
      let sigmaResult = sigmaE (tm, ty)
      in concatMap (loop sig cxt) sigmaResult
    _ -> [(tm, ty)]

sigmaE :: (DTT.Preterm, DTT.Preterm) -> [(DTT.Preterm, DTT.Preterm)]
sigmaE (m, (DTT.Sigma a b)) = [((DTT.Proj DTT.Fst m), a), ((DTT.Proj DTT.Snd m), (DTT.subst b (DTT.Proj DTT.Fst m) 0))]
