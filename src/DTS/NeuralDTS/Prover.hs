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

module DTS.NeuralDTS.Prover (
  strToEntityPred
  ) where

import qualified Data.Text.Lazy as T      --text
import qualified Data.Text.Lazy.IO as T   --text
import qualified Data.List as L           --base
import Control.Monad (when,forM_,join) 
import Control.Monad.Trans.List (ListT, runListT)
import Control.Monad.State (lift)         --mtl
import Control.Monad.IO.Class (liftIO)    --base
import qualified Data.Map as Map
import qualified Data.Maybe (mapMaybe, fromMaybe)
-- import ListT (ListT(..),fromFoldable,toList,take,null) 
import qualified Parser.ChartParser as CP
import qualified Parser.CCG as CCG 
import qualified Parser.Language.Japanese.Templates as TP
import qualified Interface.HTML as HTML
import qualified Interface.Text as T
import qualified DTS.DTTdeBruijn as DTT
import qualified DTS.DTTdeBruijn as DTTdB
import qualified DTS.UDTTdeBruijn as UDTT
import qualified DTS.UDTTdeBruijn as UDTTdB
import qualified DTS.UDTTwithName as UDTTwN
import qualified DTS.QueryTypes as QT
import qualified DTS.TypeChecker as TY
import qualified DTS.NaturalLanguageInference as NLI
import qualified DTS.NeuralDTS.Preprocessing as PP

import qualified System.IO as S
import Debug.Trace

strToEntityPred :: CP.ParseSetting -> Int ->  DTT.Signature -> DTT.Context -> [T.Text] -> IO ()
strToEntityPred ps nbest signtr contxt str = do
  -- 文に番号を振る
  let numberedStr = zipWith (\i s -> (T.pack $ "S" ++ show i, s)) [1..] str
  
  -- 文ごとに解析を行う
  nodeslist <- mapM (\(num, s) -> fmap (map (\n -> (num, n))) $ CP.simpleParse ps s) numberedStr
  
  let pairslist = map (map (\(num, node) -> (num, node, UDTT.betaReduce $ UDTT.sigmaElimination $ CP.sem node)) . take nbest) nodeslist;
      chosenlist = choice pairslist
      nodeSRlist = map unzip3 chosenlist
      nds = concat $ map (\(_, nodes, _) -> nodes) nodeSRlist
      srs = concat $ map (\(nums, _, srs) -> zip nums srs) nodeSRlist -- :: [(T.Text, UD.Preterm)]
    --   sig = foldl L.union [] $ map CP.sig nds -- :: [(T.Text,Preterm)] (= UD.Signature)
    --   sig = L.nub $ (CCG.sig (head nodeslist)) ++ signtr
      sig = L.nub $ concatMap (\(_, nodes, _) -> concatMap CCG.sig nodes) nodeSRlist ++ signtr

  -- 型チェック結果を表示する
  putStrLn $ "~~srs~~"
  printList srs 
  
  putStrLn $ "~~sig~~"
  printList sig
  
  -- let initialEnv = map snd sig
      -- judges = PP.getJudgements initialEnv [((DTTdB.Con x), y) | (x, _) <- srs, (_, y) <- srs] -- :: [([UJudgement], [UJudgement])]    
--   let judges = PP.getJudgements sig contxt [((DTTdB.Con x), y) | (x, _) <- srs, (_, y) <- srs] -- :: [([UJudgement], [UJudgement])]    
  let convertedSrs = [ (DTTdB.Con x, case UDTTdB.toDTT y of
                                       Just val -> val  -- 変換結果が Just の場合
                                       Nothing  -> DTTdB.Con("None"))  -- Nothing の場合にデフォルト値を返す
                   | (x, y) <- srs ]
      judges = PP.getJudgements sig contxt convertedSrs   
      entitiesJudges = map fst judges -- :: [[UJudgement]]   
      predsJudges = map snd judges -- :: [[UJudgement]]
      entities = map extractTermPreterm entitiesJudges -- :: [[Preterm]]
      correctPreds = map extractTypePreterm predsJudges -- :: [[Preterm]]

  let (tmp1, others) = L.partition isEntity [((DTTdB.Con x), y) | (x, y) <- sig]
      allEntities = entities ++ [map fst tmp1]
      sigPreds = map fst others
  
  -- entitiesの辞書
  let entitiesIndex = pretermsIndex allEntities
  -- entityの総数
  let entitiesNum = length entitiesIndex
  putStrLn $ "~~entityの辞書~~ "
  putStrLn $ (show entitiesIndex) ++ " " ++ (show entitiesNum) ++ "個"
  -- id->entityのマップ
  -- let entityMap = Map.fromList entitiesIdx
  -- putStrLn $ "Entity Map: " ++ show entityMap

  -- predsの辞書
  let predsIndex = pretermsIndex [sigPreds]
  -- predsの総数
  let predsNum = length predsIndex
  putStrLn $ "~~述語の辞書~~ "
  putStrLn $ (show predsIndex) ++ " " ++ (show predsNum) ++ "個"
  -- id->述語のマップ
  -- let predsIdxMap = Map.fromList indexPreds
  -- putStrLn $ "Predicate Map: " ++ show predsIdxMap
  -- putStrLn $ show entitiesNum ++ "," ++ show predsNum

--    -- 成り立つpreds
  putStrLn $ "~~成り立つ述語~~ "
  let groupedPreds = groupPredicatesByArity $ concat correctPreds
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

choice :: [[a]] -> [[a]]
choice [] = [[]]

indexPreterms :: [[DTTdB.Preterm]] -> [(Int, DTTdB.Preterm)]
indexPreterms = snd . L.foldl' addIndexedGroup (0, [])
  where
    addIndexedGroup :: (Int, [(Int, DTTdB.Preterm)]) -> [DTTdB.Preterm] -> (Int, [(Int, DTTdB.Preterm)])
    addIndexedGroup (startIndex, acc) group = 
      let indexed = zip [startIndex..] group
          newIndex = startIndex + length group
      in (newIndex, acc ++ indexed)

pretermsIndex :: [[DTTdB.Preterm]] -> [(DTTdB.Preterm, Int)]
pretermsIndex = snd . L.foldl' addIndexedGroup (0, [])
  where
    addIndexedGroup :: (Int, [(DTTdB.Preterm, Int)]) -> [DTTdB.Preterm] -> (Int, [(DTTdB.Preterm, Int)])
    addIndexedGroup (startIndex, acc) group = 
      let indexed = [(term, index) | (index, term) <- zip [startIndex..] group]
          newIndex = startIndex + length group
      in (newIndex, acc ++ indexed)

extractTermPreterm :: [DTTdB.Judgment] -> [DTTdB.Preterm]
extractTermPreterm = map (\(DTTdB.Judgment _ _ preterm _) -> preterm)
extractTypePreterm :: [DTTdB.Judgment] -> [DTTdB.Preterm]
extractTypePreterm = map (\(DTTdB.Judgment _ _ _ preterm) -> preterm)

-- printList :: [(T.Text, DTTdB.Preterm)] -> IO ()
printList [] = return ()
printList ((text, preterm):xs) = do
    T.putStr "Text: "
    T.putStrLn text
    putStr "Preterm: "
    print preterm
    printList xs
    putStr ""

isEntity :: (DTTdB.Preterm, DTTdB.Preterm) -> Bool
isEntity (_, (DTTdB.Con cname)) = cname == "entity"
isEntity _ = False

-- isPred :: (DTTdB.Preterm, DTTdB.Preterm) -> Bool
-- isPred (tm, ty) = 
--   trace ("tm : " ++ show tm ++ "ty : " ++ show ty) $
--   case ty of
--     DTTdB.App f x -> True
--     _ -> False

containsFunctionType :: DTTdB.Preterm -> Bool
containsFunctionType term = case term of
    DTTdB.Pi _ _ -> True
    DTTdB.Lam _ -> True
    DTTdB.App f x -> containsFunctionType f || containsFunctionType x
    _ -> False

groupPredicatesByArity :: [DTTdB.Preterm] -> Map.Map Int [DTTdB.Preterm]
groupPredicatesByArity predicates = 
    Map.fromListWith (++) $ groupSingle predicates
  where
    groupSingle preds = [(countArgs p, [p]) | p <- preds]

countArgs :: DTTdB.Preterm -> Int
-- countArgs term = trace ("Counting args for: " ++ show term) $ countArgsFromString (show term)
countArgs term = countArgsFromString (show term)

countArgsFromString :: String -> Int
countArgsFromString s = 
    let withoutOuterParens = T.pack $ init $ tail $ dropWhile (/= '(') s
        args = T.splitOn (T.pack ",") withoutOuterParens
    -- in trace ("Split args: " ++ show args) $
    in length args
