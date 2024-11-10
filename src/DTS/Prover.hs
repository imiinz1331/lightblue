{-# OPTIONS -Wall #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : DTS.Prover
Copyright   : Daisuke Bekki
Licence     : All right reserved
Maintainer  : Daisuke Bekki <bekki@is.ocha.ac.jp>
Stability   : beta

A Prover-Interfaces for DTS.
-}

module DTS.Prover (
  defaultTypeCheck,
  defaultProofSearch,
  checkFelicity,
  checkEntailment,
  strToEntityPred
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
import Debug.Trace

-- | type check with the default signature = entity:type, evt:type
defaultTypeCheck :: UD.Signature -> UD.Context -> UD.Preterm -> UD.Preterm -> [Ty.UTree  Ty.UJudgement]
defaultTypeCheck sig cont term typ = Ty.typeCheckU cont (("evt",UD.Type):("entity",UD.Type):sig) term typ

-- | proof search with the default signature = entity:type, evt:type
defaultProofSearch :: UD.Signature -> UD.Context -> UD.Preterm -> [Ty.UTree  Ty.UJudgement]
defaultProofSearch sig cont typ = Ty.proofSearch cont (("evt",UD.Type):("entity",UD.Type):sig) typ

-- | checks felicity condition
checkFelicity :: UD.Signature -> [UD.Preterm] -> UD.Preterm -> [Ty.UTree  Ty.UJudgement]
checkFelicity sig cont term = defaultTypeCheck sig cont term (UD.Type)

-- | executes type check to a context
sequentialTypeCheck :: UD.Signature -> [UD.Preterm] -> [UD.Preterm]
sequentialTypeCheck sig = foldr (\sr cont -> let result = do
                                                          t1 <- checkFelicity sig cont sr;
                                                          t2 <- Ty.aspElim t1;
                                                          t3 <- Ty.getTerm t2
                                                          return $ Ty.repositP t3 in
                                             if null result
                                                then (UD.Con "Typecheck or aspElim failed"):cont
                                                else (head result):cont
                                ) []

strToEntityPred :: Int -> Int -> [T.Text] -> IO ()
strToEntityPred beam nbest str = do
  -- S1, S2, ... と文に番号を振る
  let numberedStr = zipWith (\i s -> (T.pack $ "S" ++ show i, s)) [1..] str
  nodeslist <- mapM (\(num, s) -> fmap (map (\n -> (num, n))) $ CP.simpleParse beam s) numberedStr
  
  let pairslist = map (map (\(num, node) -> (num, node, UD.betaReduce $ UD.sigmaElimination $ CP.sem node)) . take nbest) nodeslist;
      chosenlist = choice pairslist
      nodeSRlist = map unzip3 chosenlist
      nds = concat $ map (\(_, nodes, _) -> nodes) nodeSRlist
      srs = concat $ map (\(nums, _, srs) -> zip nums srs) nodeSRlist -- :: [(T.Text, UD.Preterm)]
      sig = foldl L.union [] $ map CP.sig nds -- :: [(T.Text,Preterm)] (= UD.Signature)

  putStrLn $ "~~srs~~"
  printList srs 
  
  putStrLn $ "~~sig~~"
  printList sig

  -- let initialEnv = map snd sig
      -- judges = Ty.getJudgements initialEnv [((UD.Con x), y) | (x, _) <- srs, (_, y) <- srs] -- :: [([UJudgement], [UJudgement])]    
  let judges = Ty.getJudgements [] [((UD.Con x), y) | (x, _) <- srs, (_, y) <- srs] -- :: [([UJudgement], [UJudgement])]    
      entitiesJudges = map fst judges -- :: [[UJudgement]]   
      predsJudges = map snd judges -- :: [[UJudgement]]
      entities = map extractTermPreterm entitiesJudges -- :: [[Preterm]]
      preds = map extractTypePreterm predsJudges -- :: [[Preterm]]

  let (tmp1, others) = L.partition isEntity [((UD.Con x), y) | (x, y) <- sig]
      allEntities = entities ++ [map fst tmp1]
      sigPreds = map fst others

  -- mapM_ (\(num, entities) -> do
  --   putStrLn $ T.unpack num ++ " entity:"
  --   mapM_ (\j -> putStrLn $ "  " ++ show j) entitiesJudges
  --   putStrLn ""
  --   ) $ zip (map fst srs) entities
  -- mapM_ (\(num, preds) -> do
  --   putStrLn $ T.unpack num ++ " pred:"
  --   mapM_ (\j -> putStrLn $ "  " ++ show j) predsJudges
  --   putStrLn ""
  --   ) $ zip (map fst srs) preds
  
  -- entitiesの辞書
  let entitiesIndex = indexPreterms allEntities :: [(Int, UD.Preterm)]
  -- entityの総数
  let entitiesNum = length entitiesIndex
  putStrLn $ "~~entityの辞書~~ "
  putStrLn $ (show entitiesIndex) ++ " " ++ (show entitiesNum) ++ "個"
  -- id->entityのマップ
  -- let entityMap = Map.fromList entitiesIdx
  -- putStrLn $ "Entity Map: " ++ show entityMap

  -- predsの辞書
  let predsIndex = indexPreterms [sigPreds]
  -- predsの総数
  let predsNum = length predsIndex
  putStrLn $ "~~述語の辞書~~ "
  putStrLn $ (show predsIndex) ++ " " ++ (show predsNum) ++ "個"
  -- id->述語のマップ
  -- let predsIdxMap = Map.fromList indexPreds
  -- putStrLn $ "Predicate Map: " ++ show predsIdxMap
  -- putStrLn $ show entitiesNum ++ "," ++ show predsNum

  -- 成り立つpreds
  putStrLn $ "~~成り立つ述語~~ "
  putStrLn $ show preds
  -- id->述語のマップ
  -- let predsIdxMap = Map.fromList indexPreds
  -- putStrLn $ "Predicate Map: " ++ show predsIdxMap
  -- putStrLn $ show entitiesNum ++ "," ++ show predsNum

indexPreterms :: [[UD.Preterm]] -> [(Int, UD.Preterm)]
indexPreterms = snd . L.foldl' addIndexedGroup (0, [])
  where
    addIndexedGroup :: (Int, [(Int, UD.Preterm)]) -> [UD.Preterm] -> (Int, [(Int, UD.Preterm)])
    addIndexedGroup (startIndex, acc) group = 
      let indexed = zip [startIndex..] group
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

-- | checks if premises entails hypothesis
checkEntailment :: Int         -- ^ beam width
                   -> Int      -- ^ n-best
                   -> [T.Text] -- ^ premises
                   -> T.Text   -- ^ a hypothesis
                   -> IO()
checkEntailment beam nbest premises hypothesis = do
  let hline = "<hr size='15' />"
  --
  -- Show premises and hypothesis
  --
  --mapM_ T.putStr ["[", jsem_id, "]"]
  mapM_ (\p -> mapM_ T.putStr ["<p>P: ", p, "</p>"]) premises
  mapM_ T.putStr ["<p>H: ", hypothesis, "</p>"]
  T.putStrLn hline
  --
  -- Parse sentences
  --
  let sentences = hypothesis:(reverse premises)     -- reverse the order of sentences (hypothesis first, the first premise last)
  nodeslist <- mapM (CP.simpleParse beam) sentences -- parse sentences
  let pairslist = map ((map (\node -> (node, UD.betaReduce $ UD.sigmaElimination $ CP.sem node))).(take nbest)) nodeslist;
      -- Example: [[(nodeA1,srA1),(nodeA2,srA2)],[(nodeB1,srB1),(nodeB2,srB2)],[(nodeC1,srC1),(nodeC2,srC2)]]
      --          where sentences = A,B,C (where A is the hypothesis), nbest = 2_
      chosenlist = choice pairslist;
      -- Example: [[(nodeA1,srA1),(nodeB1,srB1),(nodeC1,srC1)],[(nodeA1,srA1),(nodeB1,srB1),(nodeC2,srC2)],...]
      nodeSRlist = map unzip chosenlist;
      -- Example: [([nodeA1,nodeB1,nodeC1],[srA1,srB1,srC1]),([nodeA1,nodeB1,nodeC2],[srA1,srB1,srC2]),...]
  tripledNodes <- mapM (\(nds,srs) -> do
                         let newsig = foldl L.union [] $ map CP.sig nds;
                             typecheckedSRs = sequentialTypeCheck newsig srs;
                             -- Example: u0:srA1, u1:srB1, u2:srC1 (where A1 is the hyp.)
                             -- この時点で一文目はtypecheck of aspElim failed
                             proofdiagrams = case typecheckedSRs of
                                               [] -> []
                                               (hype:prems) -> defaultProofSearch newsig prems hype;
                         return (nds,typecheckedSRs,proofdiagrams)
                       ) nodeSRlist;
  --S.hPutStrLn S.stderr $ show tripledNodes
  let nodeSrPrList = dropWhile (\(_,_,p) -> null p) tripledNodes;
      (nds,srs,pds) = if null nodeSrPrList
                        then head tripledNodes
                        else head nodeSrPrList
  --
  -- Show parse trees
  --
  T.putStrLn HTML.startMathML
  mapM_ (T.putStrLn . HTML.toMathML) $ reverse nds
  T.putStrLn HTML.endMathML
  T.putStrLn hline
  --
  -- Show proof diagrams
  --
  if null pds
     then mapM_ T.putStrLn [
           "No proof diagrams for: ",
           HTML.startMathML,
           UD.printProofSearchQuery (tail srs) (head srs),
           HTML.endMathML
           ]
      else do
           T.putStrLn "Proved: "
           T.putStrLn HTML.startMathML
           mapM_ (T.putStrLn . Ty.utreeToMathML) pds
           T.putStrLn HTML.endMathML
  T.putStrLn hline

choice :: [[a]] -> [[a]]
choice [] = [[]]
choice (a:as) = [x:xs | x <- a, xs <- choice as]
