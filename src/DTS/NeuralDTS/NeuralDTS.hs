{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module DTS.NeuralDTS.NeuralDTS (
  embedEntities
  ) where

import Control.Monad (when)
import Data.List (foldl', intersperse, scanl')
import GHC.Generics
import Torch

import qualified DTS.UDTT as UD

data MLPSpec = MLPSpec
  { feature_counts :: [Int],
    nonlinearitySpec :: Tensor -> Tensor
  }

embedEntities :: [[UD.Preterm]] -> IO [Tensor]
embedEntities allEntities = do
  -- MLPモデルの初期化
  mlpModel <- sample $ MLPSpec
    { feature_counts = [2, 4, 8], -- 適宜調整
      nonlinearitySpec = Torch.tanh
    }

  -- エンティティを埋め込む
  let entitiesTensors = map (\e -> asTensor $ map fromIntegral e) allEntities -- :: [Tensor]
      embeddedEntities = map (mlp mlpModel) entitiesTensors -- :: [Tensor]

  putStrLn "~~埋め込まれたエンティティ~~"
  mapM_ print embeddedEntities

  return embeddedEntities