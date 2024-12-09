{-# OPTIONS -Wall#-}
{-# LANGUAGE OverloadedStrings #-}

module DTS.NeuralDTS.Preprocessing
( getJudgements,
  DTTdB.Judgment(..)
) where

import Data.List (lookup)    --base
import Control.Applicative (empty)   --base
import Control.Monad (guard,when,sequence)    --base
import Control.Monad.State (lift)  
import ListT (ListT(..),fromFoldable) --list-t
import qualified System.IO as S           --base
import qualified Data.List as L           -- base
import qualified Data.Text.Lazy as T --text
import qualified Data.Text.Lazy.IO as T --text
import Interface.Text (SimpleText(..))
import Interface.Tree (Tree(..),node)
import qualified DTS.DTTdeBruijn as DTTdB
import qualified DTS.UDTTdeBruijn as UDTTdB
import qualified DTS.UDTTwithName as UDTTwN
import qualified DTS.QueryTypes as QT
import Debug.Trace

-- -- | A type of an element of a type signature, that is, a list of pairs of a preterm and a type.
-- -- ex. [entity:type, state:type, event:type, student:entity->type]
-- type Signature = [(T.Text,DTTdB.Preterm)]

-- -- TUEnv : UDTTの型環境の型
-- -- | haddock
-- type TUEnv = [DTTdB.Preterm]

-- -- TEnv : DTTの型環境の型
-- -- | haddock
-- type TEnv = [DTTdB.Preterm]

-- -- | SUEnv : UDTTのシグネチャの型
-- type SUEnv = DTTdB.Signature

-- termとtypeを受け取って([entity], [述語])のlistを得る
getJudgements :: DTTdB.Signature -> DTTdB.Context -> [(DTTdB.Preterm, DTTdB.Preterm)] -> [([DTTdB.Judgment], [DTTdB.Judgment])]
getJudgements _ _ [] = []
getJudgements signature context ((tm, ty) : rest) =
  let newPairs = loop context (tm, ty)
      newJudgements = map (\(tm2, ty2) -> DTTdB.Judgment {
          DTTdB.signtr = signature,
          DTTdB.contxt = context,
          DTTdB.trm = tm2,
          DTTdB.typ = ty2
      }) newPairs
      (entities, others) = L.partition isEntity newJudgements
      (preds, _) = L.partition isPred others
  in  ((entities, preds) : getJudgements signature context rest)
  where
      isEntity (DTTdB.Judgment _ _ _ (DTTdB.Con cname)) = cname == "entity"
      isEntity _ = False
      isPred (DTTdB.Judgment _ _ _ ty) = 
          case ty of
            DTTdB.App f x ->
                trace ("Checking isPred for: " ++ show ty) $
                not (containsFunctionType f) && 
                not (containsFunctionType x)
            _ -> False

containsFunctionType :: DTTdB.Preterm -> Bool
containsFunctionType term = case term of
    DTTdB.Pi _ _ -> trace ("Pi found in: " ++ show term) True
    DTTdB.Lam _ -> trace ("Lambda found in: " ++ show term) True
    DTTdB.App f x -> trace ("App found in: " ++ show term) $ containsFunctionType f || containsFunctionType x
    _ -> False

-- loop :: DTTdB.Context -> (DTTdB.Preterm, DTTdB.Preterm) -> [(DTTdB.Preterm, DTTdB.Preterm)]
-- loop env (tm, ty) = case ty of
--     DTTdB.Sigma _ _ ->
--       let sigmaResult = sigmaE (tm, ty)
--       in concatMap (loop env) sigmaResult
--     _ -> [(tm, ty)]

loop :: DTTdB.Context -> (DTTdB.Preterm, DTTdB.Preterm) -> [(DTTdB.Preterm, DTTdB.Preterm)]
loop env (tm, ty) = case ty of
    DTTdB.Sigma _ _ ->
      let sigmaResult = sigmaE (tm, ty)
      in trace ("Sigma Result: " ++ show sigmaResult) $
         concatMap (loop env) sigmaResult
    _ -> [(tm, ty)]

sigmaE :: (DTTdB.Preterm, DTTdB.Preterm) -> [(DTTdB.Preterm, DTTdB.Preterm)]
sigmaE (m, (DTTdB.Sigma a b)) = 
    [((DTTdB.Proj DTTdB.Fst m), a), ((DTTdB.Proj DTTdB.Snd m), (DTTdB.subst b (DTTdB.Proj DTTdB.Fst m) 0))]
sigmaE _ = []

{-
-- テスト関数
testEliminateSigma :: Bool
testEliminateSigma = and
  [ testNonDependentSigma,
    testDependentSigma,
    testWalkingManSigma
  ]

testNonDependentSigma :: Bool
testNonDependentSigma =
    let env = []
        term = DTTdB.Var 0
        sigmaType = DTTdB.Sigma (DTTdB.Con (T.pack "A")) (DTTdB.Con (T.pack "B"))
        result = loop env (term, sigmaType)
        expected = [ (DTTdB.Proj DTTdB.Fst (DTTdB.Var 0), DTTdB.Con (T.pack "A"))
                   , (DTTdB.Proj DTTdB.Snd (DTTdB.Var 0), DTTdB.Con (T.pack "B"))
                   ]
    in trace ("Non Dependen Sigma test: " ++ show (result == expected)) (result == expected)

testDependentSigma :: Bool
testDependentSigma =
    let env = []
        -- Var 0 : (x : Nat) × Vec A x
        term = DTTdB.Var 0
        sigmaType = DTTdB.Sigma 
                        DTTdB.Nat 
                        (DTTdB.App (DTTdB.Con (T.pack "Vec")) (DTTdB.Con (T.pack "A")) `DTTdB.App` DTTdB.Var 0)
        result = loop env (term, sigmaType)
        expected = [ (DTTdB.Proj DTTdB.Fst (DTTdB.Var 0), DTTdB.Nat) -- π1(Var 0) : Nat
                   , (DTTdB.Proj DTTdB.Snd (DTTdB.Var 0),  -- π2(Var 0) : Vec A (π1(Var 0))
                      DTTdB.App (DTTdB.Con (T.pack "Vec")) (DTTdB.Con (T.pack "A")) `DTTdB.App` (DTTdB.Proj DTTdB.Fst (DTTdB.Var 0)))
                   ]
    in trace ("Dependent Sigma test: " ++ show (result == expected)) (result == expected)

testWalkingManSigma :: Bool
testWalkingManSigma =
    let env = []
        -- S : (u : (x : entity × man(x))) × walk(π1(u))
        term = DTTdB.Con (T.pack "S") 
        sigmaType = DTTdB.Sigma 
                        (DTTdB.Sigma 
                            (DTTdB.Con (T.pack "entity")) 
                            (DTTdB.App (DTTdB.Con (T.pack "man")) (DTTdB.Var 0)))
                        (DTTdB.App (DTTdB.Con (T.pack "walk")) (DTTdB.Proj DTTdB.Fst (DTTdB.Var 0)))
        result = loop env (term, sigmaType)
        expected = [ 
            (DTTdB.Proj DTTdB.Fst (DTTdB.Proj DTTdB.Fst (DTTdB.Con (T.pack "S"))), DTTdB.Con (T.pack "entity")), -- π1π1S : entity
            (DTTdB.Proj DTTdB.Snd (DTTdB.Proj DTTdB.Fst (DTTdB.Con (T.pack "S"))),  -- π2π1S : man(π1π1S)
             DTTdB.App (DTTdB.Con (T.pack "man")) (DTTdB.Proj DTTdB.Fst (DTTdB.Proj DTTdB.Fst (DTTdB.Con (T.pack "S"))))),
            (DTTdB.Proj DTTdB.Snd (DTTdB.Con (T.pack "S")), -- π2S : walk(π1π1S)
             DTTdB.App (DTTdB.Con (T.pack "walk")) (DTTdB.Proj DTTdB.Fst (DTTdB.Proj DTTdB.Fst (DTTdB.Con (T.pack "S")))))
          ]
        traceResult = trace ("Result: " ++ show result) result
        traceExpected = trace ("Expected: " ++ show expected) expected
        isEqual = traceResult == traceExpected
    in trace ("WalkingMan Sigma test: " ++ show isEqual) isEqual
    
runTests = do
  let allTestsPassed = testEliminateSigma
  putStrLn $ if allTestsPassed
    then "All tests passed!"
    else "Some tests failed."

-- getJudgesのテスト関数
testGetJudges :: Bool
testGetJudges = and
  [ testManWalking,
    testManEnteredWhistled
  ]

testManWalking :: Bool
testManWalking =
    let env = []
        -- S : (u : (x : entity × man(x))) × walk(π1(u))
        term = DTTdB.Con (T.pack "S") 
        sigmaType = DTTdB.Sigma 
                        (DTTdB.Sigma 
                            (DTTdB.Con (T.pack "entity")) 
                            (DTTdB.App (DTTdB.Con (T.pack "man")) (DTTdB.Var 0)))
                        (DTTdB.App (DTTdB.Con (T.pack "walk")) (DTTdB.Proj DTTdB.Fst (DTTdB.Var 0)))
        result = getJudgements env [(term, sigmaType)]
        expected = [
            ([DTTdB.Judgment env (DTTdB.Proj DTTdB.Fst (DTTdB.Proj DTTdB.Fst (DTTdB.Con (T.pack "S")))) (DTTdB.Con (T.pack "entity"))], -- π1π1S : entity
             [DTTdB.Judgment env (DTTdB.Proj DTTdB.Snd (DTTdB.Proj DTTdB.Fst (DTTdB.Con (T.pack "S"))))  -- π2π1S : man(π1π1S)
              (DTTdB.App (DTTdB.Con (T.pack "man")) (DTTdB.Proj DTTdB.Fst (DTTdB.Proj DTTdB.Fst (DTTdB.Con (T.pack "S"))))),
              DTTdB.Judgment env (DTTdB.Proj DTTdB.Snd (DTTdB.Con (T.pack "S"))) -- π2S : walk(π1π1S)
              (DTTdB.App (DTTdB.Con (T.pack "walk")) (DTTdB.Proj DTTdB.Fst (DTTdB.Proj DTTdB.Fst (DTTdB.Con (T.pack "S")))))])
          ]
        traceResult = trace ("Result: " ++ show result) result
        traceExpected = trace ("Expected: " ++ show expected) expected
        isEqual = traceResult == traceExpected
        -- isEqual = seq (trace ("Result: " ++ show result) ()) True
    in trace ("WalkingMan test: " ++ show isEqual) isEqual

testManEnteredWhistled :: Bool
testManEnteredWhistled =
    let env = []
        -- 1文目: A man entered.
        term1 = DTTdB.Con (T.pack "S1")
        type1 = DTTdB.Sigma
                    (DTTdB.Sigma
                        (DTTdB.Con (T.pack "entity"))
                        (DTTdB.App (DTTdB.Con (T.pack "man")) (DTTdB.Var 0)))
                    (DTTdB.App (DTTdB.Con (T.pack "enter")) (DTTdB.Proj DTTdB.Fst (DTTdB.Var 0)))

        -- 2文目: He whistled.
        term2 = DTTdB.Con (T.pack "S2")
        type2 = DTTdB.App
                    (DTTdB.Con (T.pack "whistle"))
                    (DTTdB.Proj DTTdB.Fst (DTTdB.Proj DTTdB.Fst (DTTdB.Con (T.pack "S1"))))

        result = getJudgements env [(term1, type1), (term2, type2)]

        expected = [
            -- 1文目の結果
            ([DTTdB.Judgment env (DTTdB.Proj DTTdB.Fst (DTTdB.Proj DTTdB.Fst (DTTdB.Con (T.pack "S1")))) (DTTdB.Con (T.pack "entity"))],
             [DTTdB.Judgment env (DTTdB.Proj DTTdB.Snd (DTTdB.Proj DTTdB.Fst (DTTdB.Con (T.pack "S1")))) (DTTdB.App (DTTdB.Con (T.pack "man")) (DTTdB.Proj DTTdB.Fst (DTTdB.Proj DTTdB.Fst (DTTdB.Con (T.pack "S1"))))),
              DTTdB.Judgment env (DTTdB.Proj DTTdB.Snd (DTTdB.Con (T.pack "S1"))) (DTTdB.App (DTTdB.Con (T.pack "enter")) (DTTdB.Proj DTTdB.Fst (DTTdB.Proj DTTdB.Fst (DTTdB.Con (T.pack "S1")))))]),
            -- 2文目の結果
            ([],
             [DTTdB.Judgment env (DTTdB.Con (T.pack "S2")) (DTTdB.App (DTTdB.Con (T.pack "whistle")) (DTTdB.Proj DTTdB.Fst (DTTdB.Proj DTTdB.Fst (DTTdB.Con (T.pack "S1")))))])
          ]

        traceResult = trace ("Result: " ++ show result) result
        traceExpected = trace ("Expected: " ++ show expected) expected
        isEqual = traceResult == traceExpected
    in trace ("Man Entered Whistled test: " ++ show isEqual) isEqual

runTests = do
  let allTestsPassed = testGetJudges
  putStrLn $ if allTestsPassed
    then "All tests passed!"
    else "Some tests failed."
-}