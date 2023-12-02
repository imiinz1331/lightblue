{-# LANGUAGE GADTs, StandaloneDeriving, TypeSynonymInstances, FlexibleInstances #-}

{-|
Module      : DTS.NamedUDTT
Copyright   : (c) Daisuke Bekki, 2023
Licence     : All right reserved
Maintainer  : Daisuke Bekki <bekki@is.ocha.ac.jp>
Stability   : beta

Underspecified Dependent Type Theory (with variable names).
-}

module DTS.UDTTvarName (
  -- * Terms and Types
  VarName(..)
  , Selector(..)
  , Preterm(..)
  -- * Conversion btw. De Bruijn notation and Variable-name notation
  --, Indexed(..)
  --, initializeIndex
  , fromDeBruijn
  , toDeBruijn
  , fromDeBruijnSignature
  , fromDeBruijnList
  -- * Judgment
  , Signature
  , Context
  , Judgment(..)
  , toVerticalMathML
  , printVerticalMathML
  ) where

import qualified Data.Text.Lazy as T    -- text
import qualified Data.Text.Lazy.IO as T -- text
import qualified Data.List as L           -- base
import qualified Control.Applicative as M -- base
import qualified Control.Monad as M       -- base
import Interface.Text
import Interface.TeX
import Interface.HTML
import qualified DTS.UDTTdeBruijn as DB

-- | A variable name consists of Char (e.g. 'x') and Int (e.g. 1), which is displayed as $x_{1}$ in TeX and $x1$ in Text.
data VarName = VarName Char Int deriving (Eq,Show)

instance SimpleText VarName where
  toText (VarName v i) = T.cons v $ T.pack (show i)
instance Typeset VarName where
  toTeX (VarName v i) = T.cons v $ T.concat ["_{", T.pack (show i), "}"]
instance MathML VarName where
  toMathML (VarName v i) = T.concat ["<msub><mi>", T.singleton v, "</mi><mn>", T.pack (show i), "</mn></msub>"]

-- | 'Proj' 'Fst' m is the first projection of m, while 'Proj' 'Snd' m is the second projection of m.
data Selector = Fst | Snd
  deriving (Eq, Show)

-- | print a selector as "1" or "2".
instance SimpleText Selector where
  toText Fst = "1"  -- `Proj` `Fst` m is the first projection of m
  toText Snd = "2" -- `Proj` `Snd` m is the second projection of m
instance Typeset Selector where
  toTeX = toText
instance MathML Selector where
  toMathML Fst = "<mn>1</mn>"  -- `Proj` `Fst` m is the first projection of m
  toMathML Snd = "<mn>2</mn>" -- `Proj` `Snd` m is the second projection of m

-- | Preterms of Underspecified Dependent Type Theory (UDTT) with variable names
data Preterm a where
  -- | Basic Preterms
  Var :: VarName -> Preterm a       -- ^ Variables
  Con :: T.Text -> Preterm a        -- ^ Constant symbols
  Type :: Preterm a                 -- ^ The sort \"type\"
  Kind :: Preterm a                 -- ^ The sort \"kind\"
  -- | Pi Types
  Pi :: VarName -> Preterm a -> Preterm a -> Preterm a -- ^ Pi types
  Lam :: VarName -> Preterm a -> Preterm a             -- ^ Lambda abstractions
  App :: Preterm a -> Preterm a -> Preterm a           -- ^ Function Applications
  Not :: Preterm a -> Preterm a                        -- ^ Negations
  -- | Sigma Types
  Sigma :: VarName -> Preterm a -> Preterm a -> Preterm a -- ^ Sigma types
  Pair :: Preterm a -> Preterm a -> Preterm a             -- ^ Pair
  Proj :: Selector -> Preterm a -> Preterm a -- ^ (First and second) Projections
  -- | UDTT expansions
  Asp :: Int -> Preterm DB.UDTT -> Preterm DB.UDTT -- ^ Underspesified types
  Lamvec :: VarName -> Preterm DB.UDTT -> Preterm DB.UDTT   -- ^ Variable-length lambda abstraction
  Appvec :: VarName -> Preterm DB.UDTT -> Preterm DB.UDTT   -- ^ Variable-length function application
  -- | Enumeration Types
  Unit :: Preterm a                -- ^ The unit term (of type Top)
  Top :: Preterm a                 -- ^ The top type
  Bot :: Preterm a                 -- ^ The bottom type
  -- | Natural Number Types
  Nat :: Preterm a                 -- ^ The natural number type (Nat)
  Zero :: Preterm a                -- ^ 0 (of type Nat)
  Succ :: Preterm a -> Preterm a   -- ^ The successor function
  Natrec :: Preterm a -> Preterm a -> Preterm a -> Preterm a -- ^ natrec
  -- | Intensional Equality Types
  Eq :: Preterm a -> Preterm a -> Preterm a -> Preterm a -- ^ Intensional equality type
  Refl :: Preterm a -> Preterm a -> Preterm a            -- ^ refl
  Idpeel :: Preterm a -> Preterm a -> Preterm a          -- ^ idpeel
  -- | ToDo: add Disjoint Union Types
  -- | ToDo: add First Universe

deriving instance Eq a => Eq (Preterm a)

{- Printing of Preterms -}

instance Show (Preterm a) where
  show = T.unpack . toText

instance SimpleText (Preterm a) where
  toText preterm = case preterm of
    Var vname -> toText vname
    Con cname -> cname
    Type -> "type"
    Kind -> "kind"
    Pi vname a b -> case b of
                      Bot -> T.concat["¬", toText a]
                      b' -> T.concat ["(", toText vname, ":", toText a, ")→ ", toText b']
    Lam vname m -> T.concat ["λ", toText vname, ".", toText m]
    App (App (Con cname) y) x ->
      T.concat [cname, "(", toText x, ",", toText y,")"]
    App (App (App (Con cname) z) y) x ->
      T.concat [cname, "(", toText x, ",", toText y,",",toText z,")"]
    App (App (App (App (Con cname) u) z) y) x ->
      T.concat [cname, "(", toText x, ",", toText y,",",toText z,",", toText u, ")"]
    App m n -> T.concat [toText m, "(", toText n, ")"]
    Sigma vname a b -> case b of
                         Top -> T.concat ["(", toText a, ")"]
                         _   -> T.concat ["(", toText vname, ":", toText a, ")× ", toText b]
    Pair m n  -> T.concat ["(", toText m, ",", toText n, ")"]
    Proj s m  -> T.concat ["π", toText s, "(", toText m, ")"]
    Lamvec vname m  -> T.concat ["λ", toText vname, "+.", toText m]
    Appvec vname m -> T.concat ["(", toText m, " ", toText vname, "+)"]
    Unit       -> "()"
    Top        -> "T"
    Bot        -> "⊥"
    Asp j m    -> T.concat["@", T.pack (show j), ":", toText m]
    Nat    -> "N"
    Zero   -> "0"
    Succ n -> T.concat ["s", toText n]
    Natrec n e f -> T.concat ["natrec(", toText n, ",", toText e, ",", toText f, ")"]
    Eq a m n -> T.concat [toText m, "=[", toText a, "]", toText n]
    Refl a m -> T.concat ["refl", toText a, "(", toText m, ")"]
    Idpeel m n -> T.concat ["idpeel(", toText m, ",", toText n, ")"]

-- | Each `Preterm` is translated by the `toTeX` method into a representation \"with variable names\" in a TeX source code.
instance Typeset (Preterm a) where
  toTeX preterm = case preterm of
    Var vname -> toTeX vname
    Con c -> T.concat["\\pred{", T.replace "~" "\\~{}" c, "}"]
    Type  -> "\\type{type}"
    Kind  -> "\\type{kind}"
    Pi vname a b -> case b of
                      Bot -> T.concat["\\neg ", toTeXEmbedded a]
                      b' -> T.concat["\\dPi[", toTeX vname , "]{", toTeX a, "}{", toTeX b, "}"]
    Lam vname m  -> T.concat["\\LAM[", toTeX vname, "]", toTeX m]
    (App (App (Con c) y) x) -> T.concat ["\\APP{", toTeXEmbedded (Con c), "}{\\left(", toTeX x, ",", toTeX y, "\\right)}" ]
    (App (App (App (Con c) z) y) x) -> T.concat ["\\APP{", toTeXEmbedded (Con c), "}{\\left(", toTeX x, ",", toTeX y, ",", toTeX z, "\\right)}" ]
    (App (App (App (App (Con c) u) z) y) x) -> T.concat ["\\APP{", toTeXEmbedded (Con c), "}{\\left(", toTeX x, ",", toTeX y, ",", toTeX z, ",", toTeX u, "\\right)}" ]
    App m n -> case n of
                 (Var _) -> T.concat ["\\APP{", toTeXEmbedded m, "}{(", toTeX n, ")}"]
                 (Con _) -> T.concat ["\\APP{", toTeXEmbedded m, "}{(", toTeX n, ")}"]
                 (Proj _ _) -> T.concat ["\\APP{", toTeXEmbedded m, "}{\\left(", toTeX n, "\\right)}"]
                 (Asp _ _) -> T.concat ["\\APP{", toTeXEmbedded m, "}{\\left(", toTeX n,"\\right)}"]
                 _ -> T.concat["\\APP{", toTeXEmbedded m, "}{", toTeXEmbedded n, "}"]
    Sigma vname a b -> case b of
                         Top -> toTeX a
                         _   -> T.concat["\\dSigma[", toTeX vname, "]{", toTeX a, "}{", toTeX b, "}"]
    Pair m n  -> T.concat["\\left(", toTeX m, ",", toTeX n, "\\right)"]
    Proj s m  -> T.concat["\\pi_", toTeX s, "\\left(", toTeX m, "\\right)"]
    Lamvec vname m   -> T.concat ["\\lambda\\vec{", toTeX vname, "}.", toTeX m]
    Appvec vname m -> T.concat ["\\APP{", toTeXEmbedded m, "}{\\vec{", toTeX vname, "}}"]
    Unit      -> "()"
    Top       -> "\\top"
    Bot       -> "\\bot"
    Asp j m   -> T.concat ["@_{", T.pack (show j), "}:", toTeX m]
    Nat       -> "\\Set{N}"
    Zero      -> "0"
    Succ n    -> T.concat ["\\type{s}", toTeX n]
    Natrec n e f -> T.concat ["\\type{natrec}\\left(", toTeX n, ",", toTeX e, ",", toTeX f, "\\right)"]
    Eq a m n  -> T.concat [toTeX m, "=_{", toTeX a,"}", toTeX n]
    Refl a m  -> T.concat ["\\type{refl}_{", toTeX a, "}\\left(", toTeX m,"\\right)"]
    Idpeel m n -> T.concat ["\\type{idpeel}\\left(", toTeX m, ",", toTeX n, "\\right)"]

toTeXEmbedded :: (Preterm a) -> T.Text
toTeXEmbedded preterm = case preterm of
  Lam vname m -> T.concat["\\left(\\LAM[", toTeX vname, "]", toTeX m, "\\right)"]
  Lamvec vname m -> T.concat ["\\left(\\lambda\\vec{", toTeX vname, "}.", toTeX m, "\\right)"]
  m          -> toTeX m

instance MathML (Preterm a) where
  toMathML preterm = case preterm of
    Var vname -> toMathML vname
    Con cname -> T.concat ["<mtext>", cname, "</mtext>"]
    Type -> "<mi>type</mi>"
    Kind -> "<mi>kind</mi>"
    Pi vname a b -> case b of
      Bot -> T.concat["<mrow><mi>&not;</mi>", toMathML a, "</mrow>"]
      b' -> T.concat ["<mrow><mfenced separators=':'>", toMathML vname, toMathML a, "</mfenced><mo>&rarr;</mo>", toMathML b, "</mrow>"]
    Lam vname m -> T.concat ["<mrow><mi>&lambda;</mi>", toMathML vname, "<mpadded lspace='-0.2em' width='-0.2em'><mo>.</mo></mpadded>", toMathML m, "</mrow>"]
    App (App (Con cname) y) x ->
      T.concat ["<mrow><mtext>", cname, "</mtext><mfenced>", toMathML x, toMathML y,"</mfenced></mrow>"]
    App (App (App (Con cname) z) y) x ->
      T.concat ["<mrow><mtext>", cname, "</mtext><mfenced>", toMathML x, toMathML y,toMathML z,"</mfenced></mrow>"]
    App (App (App (App (Con cname) u) z) y) x ->
      T.concat ["<mrow><mtext>", cname, "</mtext><mfenced>", toMathML x, toMathML y, toMathML z, toMathML u, "</mfenced></mrow>"]
    App m n -> T.concat ["<mrow>", toMathML m, "<mfenced>", toMathML n, "</mfenced></mrow>"]
    Sigma vname a b -> case b of
                         Top -> toMathML a
                         _   -> T.concat ["<mfenced open='[' close=']'><mtable columnalign='left'><mtr><mtd>", toMathML vname, "<mo>:</mo>", toMathML a, "</mtd></mtr><mtr><mtd><mpadded height='-0.5em'>", toMathML b, "</mpadded></mtd></mtr></mtable></mfenced>"]
    Pair m n  -> T.concat ["<mfenced>", toMathML m, toMathML n, "</mfenced>"]
    Proj s m  -> T.concat ["<mrow><msub><mi>&pi;</mi>", toMathML s, "</msub><mfenced>", toMathML m, "</mfenced></mrow>"]
    Lamvec vname m  -> T.concat ["<mrow><mi>&lambda;</mi><mover>", toMathML vname, "<mo>&rarr;</mo></mover><mo>.</mo>", toMathML m, "</mrow>"]
    Appvec vname m -> T.concat ["<mfenced separators=''>", toMathML m, "<mover>", toMathML vname, "<mo>&rarr;</mo></mover></mfenced>"]
    Unit       -> "<mi>()</mi>"
    Top        -> "<mi>&top;</mi>"
    Bot        -> "<mi>&bot;</mi>"
    Asp j m    -> T.concat["<mrow><msub><mo>@</mo><mn>", T.pack (show j), "</mn></msub><mo>::</mo>", toMathML m, "</mrow>"]
    Nat    -> "<mi>N</mi>"
    Zero   -> "<mi>0</mi>"
    Succ n -> T.concat ["<mrow><mi>s</mi>", toMathML n, "</mrow>"]
    Natrec n e f -> T.concat ["<mrow><mi>natrec</mi><mfenced>", toMathML n, toMathML e, toMathML f, "</mfenced></mrow>"]
    Eq a m n -> T.concat ["<mrow>", toMathML m, "<msub><mo>=</mo>", toMathML a, "</msub>", toMathML n, "</mrow>"]
    Refl a m -> T.concat ["<mrow><mi>refl</mi>", toMathML a, "<mfenced>", toMathML m, "</mfenced></mrow>"]
    Idpeel m n -> T.concat ["<mrow><mi>idpeel</mi><mfenced>", toMathML m, toMathML n, "</mfenced></mrow>"]

{- Initializing or Re-indexing of vars -}

-- | Indexed monad controls indices to be attached to preterms.  Arguments correspond to:
--   u for variables for propositions
-- | x for variables for entities
-- | e for variables for eventualities
-- | indices for asp-operators
-- | indices for DReL operators
newtype Indexed a = Indexed { indexing :: Int -> Int -> Int -> Int -> Int -> (a,Int,Int,Int,Int,Int) }

instance Monad Indexed where
  return m = Indexed (\s u x e a -> (m,s,u,x,e,a))
  (Indexed m) >>= f = Indexed (\s u x e a -> let (m',s',u',x',e',a') = m s u x e a;
                                                   (Indexed n) = f m';
                                               in
                                               n s' u' x' e' a')

instance Functor Indexed where
  fmap = M.liftM

instance M.Applicative Indexed where
  pure = return
  (<*>) = M.ap

-- | A sequential number for variable names (i.e. x_1, x_2, ...) in a context
sIndex :: Indexed Int
sIndex = Indexed (\s u x e a -> (s,s+1,u,x,e,a))

uIndex :: Indexed Int
uIndex = Indexed (\s u x e a -> (u,s,u+1,x,e,a))

xIndex :: Indexed Int
xIndex = Indexed (\s u x e a -> (x,s,u,x+1,e,a))

eIndex :: Indexed Int
eIndex = Indexed (\s u x e a -> (e,s,u,x,e+1,a))

aspIndex :: Indexed Int
aspIndex = Indexed (\s u x e a -> (e,s,u,x,e,a+1))

-- | re-assigns sequential indices to all asperands that appear in a given preterm.
initializeIndex :: Indexed a -> a
initializeIndex (Indexed m) = let (m',_,_,_,_,_) = m 0 0 0 0 0 in m'

-- | translates a preterm in de Bruijn notation into a preterm with variable name.
fromDeBruijn :: DB.Preterm a -> Preterm a
fromDeBruijn = initializeIndex . (fromDeBruijnLoop [])

fromDeBruijnLoop :: [VarName] -- ^ A context (= a list of variable names)
                    -> DB.Preterm a  -- ^ A preterm in de Bruijn notation
                    -> Indexed (Preterm a) -- ^ A preterm with variable names
fromDeBruijnLoop vnames preterm = case preterm of
  DB.Var j -> if j < length vnames
                      then return $ Var (vnames!!j)
                      else return $ Con $ T.concat ["error: var ",T.pack (show j), " in ", T.pack (show vnames)]
  DB.Con cname -> return $ Con cname
  DB.Type -> return Type
  DB.Kind -> return Kind
  DB.Pi a b -> do
    vname <- variableNameFor a
    a' <- fromDeBruijnLoop vnames a
    b' <- fromDeBruijnLoop (vname:vnames) b
    return $ Pi vname a' b'
  DB.Lam m   -> do
    i <- xIndex
    let vname = case m of
                  DB.Sigma _ _ -> VarName 'x' i
                  DB.Pi _ _    -> VarName 'x' i
                  _         -> VarName 'x' i
    m' <- fromDeBruijnLoop (vname:vnames) m
    return $ Lam vname m'
  DB.App m n -> do
    m' <- fromDeBruijnLoop vnames m
    n' <- fromDeBruijnLoop vnames n
    return $ App m' n'
  DB.Not a   -> do
    a' <- fromDeBruijnLoop vnames a
    return $ Not a'
  DB.Sigma a b -> do
    vname <- variableNameFor a
    a' <- fromDeBruijnLoop vnames a
    b' <- fromDeBruijnLoop (vname:vnames) b
    return $ Sigma vname a' b'
  DB.Pair m n  -> do
    m' <- fromDeBruijnLoop vnames m
    n' <- fromDeBruijnLoop vnames n
    return $ Pair m' n'
  DB.Proj s m  -> do
    m' <- fromDeBruijnLoop vnames m
    return $ case s of
               DB.Fst -> Proj Fst m'
               DB.Snd -> Proj Snd m'
  DB.Lamvec m  -> do
    i <- xIndex
    let vname = VarName 'x' i
    m' <- fromDeBruijnLoop (vname:vnames) m
    return $ Lamvec vname m'
  DB.Appvec j m -> do
    let vname = vnames!!j
    m' <- fromDeBruijnLoop vnames m
    return $ Appvec vname m'
  DB.Unit    -> return Unit
  DB.Top     -> return Top
  DB.Bot     -> return Bot
  DB.Asp _ m -> do
    j' <- aspIndex
    m' <- fromDeBruijnLoop vnames m
    return $ Asp j' m'
  DB.Nat    -> return Nat
  DB.Zero   -> return Zero
  DB.Succ n -> do
    n' <- fromDeBruijnLoop vnames n
    return $ Succ n'
  DB.Natrec n e f -> do
    n' <- fromDeBruijnLoop vnames n
    e' <- fromDeBruijnLoop vnames e
    f' <- fromDeBruijnLoop vnames f
    return $ Natrec n' e' f'
  DB.Eq a m n -> do
    a' <- fromDeBruijnLoop vnames a
    m' <- fromDeBruijnLoop vnames m
    n' <- fromDeBruijnLoop vnames n
    return $ Eq a' m' n'
  DB.Refl a m -> do
    a' <- fromDeBruijnLoop vnames a
    m' <- fromDeBruijnLoop vnames m
    return $ Refl a' m'
  DB.Idpeel m n -> do
    m' <- fromDeBruijnLoop vnames m
    n' <- fromDeBruijnLoop vnames n
    return $ Idpeel m' n'

variableNameFor :: DB.Preterm a -> Indexed VarName
variableNameFor preterm =
  case preterm of
    DB.Con cname | cname == "entity" -> do i <- xIndex; return $ VarName 'x' i
              | cname == "evt"    -> do i <- eIndex; return $ VarName 'e' i
              -- cname == "state"  -> VN.VarName 's' i
    DB.Eq _ _ _ -> do i <- xIndex; return $ VarName 's' i
    DB.Nat      -> do i <- xIndex; return $ VarName 'k' i
    _        -> do i <- uIndex; return $ VarName 'u' i

-- | translates a preterm with variable name into a preterm in de Bruijn notation.
toDeBruijn :: [VarName]  -- ^ A context (= a list of variable names)
              -> Preterm a -- ^ A preterm with variable names
              -> DB.Preterm a   -- ^ A preterm in de Bruijn notation
toDeBruijn vnames preterm = case preterm of
  Var vname -> case L.elemIndex vname vnames of
                    Just i -> DB.Var i
                    Nothing -> DB.Con "Error: vname not found in toDeBruijn Var"
  Con cname -> DB.Con cname
  Type -> DB.Type
  Kind -> DB.Kind
  Pi vname a b -> DB.Pi (toDeBruijn (vname:vnames) a) (toDeBruijn (vname:vnames) b)
  Lam vname m -> DB.Lam (toDeBruijn (vname:vnames) m)
  App m n -> DB.App (toDeBruijn vnames m) (toDeBruijn vnames n)
  Not a -> DB.Not (toDeBruijn vnames a)
  Sigma vname a b -> DB.Sigma (toDeBruijn (vname:vnames) a) (toDeBruijn (vname:vnames) b)
  Pair m n -> DB.Pair (toDeBruijn vnames m) (toDeBruijn vnames n)
  Proj s m -> case s of
                   Fst -> DB.Proj DB.Fst (toDeBruijn vnames m)
                   Snd -> DB.Proj DB.Snd (toDeBruijn vnames m)
  Lamvec vname m -> DB.Lamvec (toDeBruijn (vname:vnames) m)
  Appvec vname m -> case L.elemIndex vname vnames of
                        Just i -> DB.Appvec i (toDeBruijn vnames m)
                        Nothing -> DB.Con "Error: vname not found in toDeBruijn Appvec"
  Unit -> DB.Unit
  Top -> DB.Top
  Bot -> DB.Bot
  Asp i m -> DB.Asp i (toDeBruijn vnames m)
  Nat -> DB.Nat
  Zero -> DB.Zero
  Succ n -> DB.Succ (toDeBruijn vnames n)
  Natrec n e f -> DB.Natrec (toDeBruijn vnames n) (toDeBruijn vnames e) (toDeBruijn vnames f)
  Eq a m n -> DB.Eq (toDeBruijn vnames a) (toDeBruijn vnames m) (toDeBruijn vnames n)
  Refl a m -> DB.Refl (toDeBruijn vnames a) (toDeBruijn vnames m)
  Idpeel m n -> DB.Idpeel (toDeBruijn vnames m) (toDeBruijn vnames n)

{- Judgment -}

type Signature = [(T.Text, Preterm DB.DTT)]

instance SimpleText Signature where
  toText sigs = T.concat ["[", (T.intercalate ", " $ map (\(cname,ty) -> T.concat $ [toText $ Con cname, ":", toText ty]) sigs), "]"]
instance Typeset Signature where
  toTeX _ = T.empty
instance MathML Signature where
  toMathML _ = T.empty

fromDeBruijnSignature :: DB.Signature -> Signature
fromDeBruijnSignature = map (\(cname, ty) -> (cname, fromDeBruijn ty))

-- | A context is a list of pairs of a variable and a preterm.
type Context = [(VarName, Preterm DB.DTT)]

instance SimpleText Context where
  toText = (T.intercalate ", ") . (map (\(nm,tm) -> T.concat [toText nm, ":", toText tm])) . reverse
instance Typeset Context where
  toTeX = (T.intercalate ",") . (map (\(nm,tm) -> T.concat [toTeX nm, ":", toTeX tm])) . reverse
instance MathML Context where
  toMathML cont = T.concat $ ["<mfenced separators=',' open='' close=''>"] ++ (map (\(nm,tm) -> T.concat ["<mrow>", toMathML nm, "<mo>:</mo>", toMathML tm, "</mrow>"]) $ reverse cont) ++ ["</mfenced>"]

-- | As list of semantic representations (the first element is for the first sentence)
fromDeBruijnList :: [DB.Preterm a] -> [(VarName, Preterm a)]
fromDeBruijnList = initializeIndex . (fromDeBruijnListLoop [])

-- | the internal function of the fromDeBruijnSRlist function
fromDeBruijnListLoop :: [VarName] -> [DB.Preterm a] -> Indexed ([(VarName, Preterm a)])
fromDeBruijnListLoop _ [] = return []
fromDeBruijnListLoop varnames (x:xs) = do
  i <- sIndex
  let varname = VarName 's' i
  ix <- fromDeBruijnLoop varnames x
  ixs <- fromDeBruijnListLoop (varname:varnames) xs
  return ((varname,ix):ixs)

-- | the internal function of the fromDeBruijnContext function
fromDeBruijnContextLoop :: DB.Context -> Indexed ([VarName], Context)
fromDeBruijnContextLoop [] = return ([],[])
fromDeBruijnContextLoop (x:xs) = do
  (varnames,ixs) <- fromDeBruijnContextLoop xs
  i <- sIndex
  let varname = VarName 's' i
  ix <- fromDeBruijnLoop varnames x
  return $ (varname:varnames,(varname,ix):ixs)

-- | prints a context vertically.
toVerticalMathML :: Context -> T.Text
toVerticalMathML cont = T.concat $ ["<mtable columnalign='left'>"] ++ (map (\(nm,tm) -> T.concat ["<mtr><mtd>", toMathML nm, "<mo>:</mo>", toMathML tm, "</mtd></mtr>"]) $ reverse cont) ++ ["</mtable>"]

-- | prints an SR list vertically.
printVerticalMathML :: [(VarName, Preterm a)] -> IO()
printVerticalMathML cont = do
  T.putStr "<mtable columnalign='left'>"
  mapM_ (\(nm,tm) -> mapM_ T.putStr ["<mtr><mtd>",
                                     toMathML nm,
                                     "<mo>:</mo>",
                                     toMathML tm,
                                     "</mtd></mtr>"
                                     ]) cont
  T.putStrLn "</mtable>"

-- | prints a context in vertical manner.
--toVerticalMathML :: Context -> T.Text
--toVerticalMathML = toVerticalMathML . fromDeBruijnContext

-- | prints an SR list in vertical manner.
--printVerticalMathML :: [Preterm] -> IO()
--printVerticalMathML = printVerticalMathML . fromDeBruijnSRlist

-- | The data type for a judgment
data Judgment a = Judgment {
  sig :: Signature         -- ^ A signature
  , context :: Context  -- ^ A context \Gamma in \Gamma \vdash M:A
  , term :: Preterm a      -- ^ A term M in \Gamma \vdash M:A
  , typ :: Preterm DB.DTT     -- ^ A type A in \Gamma \vdash M:A
  } deriving (Eq)

instance SimpleText (Judgment a) where
  toText j = T.concat [toText $ context j, " |- ", toText $ term j, ":", toText $ typ j]
instance Typeset (Judgment a) where
  toTeX j = T.concat [toTeX $ context j, "{\\vdash}", toTeX $ term j, "{:}", toTeX $ typ j]
instance MathML (Judgment a) where
  toMathML j = T.concat ["<mrow>", toMathML $ context j, "<mo>&vdash;</mo>", toMathML $ term j, "<mo>:</mo>", toMathML $ typ j, "</mrow>"]

-- | translates a judgment in de Bruijn notation into one with variable names
fromDeBruijnJudgment :: DB.Judgment a -> Judgment a
fromDeBruijnJudgment judgment =
  let (vcontext', vterm', vtyp')
        = initializeIndex $ do
                            (varnames,vcontext) <- fromDeBruijnContextLoop $ DB.context judgment
                            vterm <- fromDeBruijnLoop varnames (DB.term judgment)
                            vtyp <- fromDeBruijnLoop varnames (DB.typ judgment)
                            return (vcontext, vterm, vtyp)
  in Judgment { sig = fromDeBruijnSignature $ DB.sig judgment
              , context = vcontext'
              , term = vterm'
              , typ = vtyp'
              }

{-
-- | prints a proof search query in MathML
printProofSearchQuery :: Context DB.DTT -> Preterm a -> T.Text
printProofSearchQuery cont ty =
  let (vcontext', vtyp')
        = initializeIndex $ do
                            (varnames,vcontext) <- fromDeBruijnContextLoop cont
                            vtyp <- fromDeBruijnLoop varnames ty
                            return (vcontext, vtyp)
  in T.concat ["<mrow>", toMathML vcontext', "<mo>&vdash;</mo><mo>?</mo><mo>:</mo>", toMathML vtyp', "</mrow>"]
-}
