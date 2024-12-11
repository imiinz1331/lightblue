{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}

module DTS.NeuralDTS.Classifier where
import GHC.Generics
import Torch
import Data.Binary

class Classifier n where
  classify :: n -> RuntimeMode -> Tensor -> Tensor -> Tensor -> Tensor


class TwoPlaceClassifier n where
  classifyTwoPlace :: n -> Tensor -> Tensor -> Tensor -> Tensor


class OnePlaceClassifier n where
  classifyOnePlace :: n -> Tensor -> Tensor -> Tensor

deriving instance Generic DeviceType
deriving instance Binary DeviceType
deriving instance Generic Device
deriving instance Binary Device