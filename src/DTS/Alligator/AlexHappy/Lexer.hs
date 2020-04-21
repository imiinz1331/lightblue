{-# OPTIONS_GHC -fno-warn-unused-binds -fno-warn-missing-signatures #-}
{-# LANGUAGE CPP #-}
{-# LINE 1 "DTS/Alligator/AlexHappy/Lexer.x" #-}

module DTS.Alligator.AlexHappy.Lexer (
  Token(..),
  scanTokens
) where

import DTS.Alligator.AlexHappy.Syntax
import DTS.Alligator.Arrowterm

#if __GLASGOW_HASKELL__ >= 603
#include "ghcconfig.h"
#elif defined(__GLASGOW_HASKELL__)
#include "config.h"
#endif
#if __GLASGOW_HASKELL__ >= 503
import Data.Array
import Data.Array.Base (unsafeAt)
#else
import Array
#endif
{-# LINE 1 "templates/wrappers.hs" #-}
-- -----------------------------------------------------------------------------
-- Alex wrapper code.
--
-- This code is in the PUBLIC DOMAIN; you may copy it freely and use
-- it for any purpose whatsoever.






import Data.Word (Word8)
















import Data.Char (ord)
import qualified Data.Bits

-- | Encode a Haskell String to a list of Word8 values, in UTF8 format.
utf8Encode :: Char -> [Word8]
utf8Encode = map fromIntegral . go . ord
 where
  go oc
   | oc <= 0x7f       = [oc]

   | oc <= 0x7ff      = [ 0xc0 + (oc `Data.Bits.shiftR` 6)
                        , 0x80 + oc Data.Bits..&. 0x3f
                        ]

   | oc <= 0xffff     = [ 0xe0 + (oc `Data.Bits.shiftR` 12)
                        , 0x80 + ((oc `Data.Bits.shiftR` 6) Data.Bits..&. 0x3f)
                        , 0x80 + oc Data.Bits..&. 0x3f
                        ]
   | otherwise        = [ 0xf0 + (oc `Data.Bits.shiftR` 18)
                        , 0x80 + ((oc `Data.Bits.shiftR` 12) Data.Bits..&. 0x3f)
                        , 0x80 + ((oc `Data.Bits.shiftR` 6) Data.Bits..&. 0x3f)
                        , 0x80 + oc Data.Bits..&. 0x3f
                        ]



type Byte = Word8

-- -----------------------------------------------------------------------------
-- The input type
















































































-- -----------------------------------------------------------------------------
-- Token positions

-- `Posn' records the location of a token in the input text.  It has three
-- fields: the address (number of chacaters preceding the token), line number
-- and column of a token within the file. `start_pos' gives the position of the
-- start of the file and `eof_pos' a standard encoding for the end of file.
-- `move_pos' calculates the new position after traversing a given character,
-- assuming the usual eight character tab stops.














-- -----------------------------------------------------------------------------
-- Default monad
















































































































-- -----------------------------------------------------------------------------
-- Monad (with ByteString input)







































































































-- -----------------------------------------------------------------------------
-- Basic wrapper


type AlexInput = (Char,[Byte],String)

alexInputPrevChar :: AlexInput -> Char
alexInputPrevChar (c,_,_) = c

-- alexScanTokens :: String -> [token]
alexScanTokens str = go ('\n',[],str)
  where go inp__@(_,_bs,s) =
          case alexScan inp__ 0 of
                AlexEOF -> []
                AlexError _ -> error "lexical error"
                AlexSkip  inp__' _ln     -> go inp__'
                AlexToken inp__' len act -> act (take len s) : go inp__'

alexGetByte :: AlexInput -> Maybe (Byte,AlexInput)
alexGetByte (c,(b:bs),s) = Just (b,(c,bs,s))
alexGetByte (_,[],[])    = Nothing
alexGetByte (_,[],(c:s)) = case utf8Encode c of
                             (b:bs) -> Just (b, (c, bs, s))
                             [] -> Nothing



-- -----------------------------------------------------------------------------
-- Basic wrapper, ByteString version
































-- -----------------------------------------------------------------------------
-- Posn wrapper

-- Adds text positions to the basic model.













-- -----------------------------------------------------------------------------
-- Posn wrapper, ByteString version














-- -----------------------------------------------------------------------------
-- GScan wrapper

-- For compatibility with previous versions of Alex, and because we can.














alex_tab_size :: Int
alex_tab_size = 8
alex_base :: Array Int Int
alex_base = listArray (0 :: Int, 57)
  [ -8
  , -109
  , -94
  , 86
  , -93
  , -96
  , -87
  , -88
  , -92
  , -89
  , 112
  , -86
  , -95
  , -101
  , -81
  , -91
  , -14
  , -83
  , -12
  , 98
  , -58
  , -79
  , 54
  , -32
  , -20
  , -40
  , 203
  , 208
  , 215
  , 0
  , 0
  , 0
  , 184
  , 0
  , 244
  , 337
  , 430
  , 0
  , 0
  , 0
  , 0
  , -19
  , 523
  , 616
  , 205
  , 0
  , 0
  , 0
  , 204
  , 0
  , 709
  , 803
  , 896
  , 989
  , 1082
  , 1175
  , 0
  , 75
  ]

alex_table :: Array Int Int
alex_table = listArray (0 :: Int, 1430)
  [ 0
  , 27
  , 26
  , 27
  , 27
  , 27
  , 45
  , 46
  , 1
  , 24
  , 23
  , 6
  , 21
  , 7
  , 14
  , 15
  , 12
  , 11
  , 10
  , 18
  , 19
  , 8
  , 37
  , 16
  , 28
  , 52
  , 52
  , 57
  , 52
  , 48
  , 29
  , 52
  , 31
  , 30
  , 52
  , 52
  , 40
  , 52
  , 41
  , 52
  , 43
  , 43
  , 43
  , 43
  , 43
  , 43
  , 43
  , 43
  , 43
  , 43
  , 49
  , 52
  , 50
  , 54
  , 52
  , 52
  , 9
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 4
  , 52
  , 22
  , 52
  , 2
  , 52
  , 52
  , 52
  , 52
  , 52
  , 51
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 33
  , 52
  , 32
  , 55
  , 55
  , 56
  , 55
  , 0
  , 5
  , 55
  , 0
  , 0
  , 55
  , 55
  , 19
  , 55
  , 13
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 10
  , 55
  , 55
  , 55
  , 55
  , 55
  , 0
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 0
  , 55
  , 0
  , 55
  , 0
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 20
  , 55
  , 27
  , 27
  , 27
  , 27
  , 27
  , 27
  , 27
  , 27
  , 27
  , 27
  , 39
  , 17
  , 27
  , 27
  , 27
  , 27
  , 27
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 27
  , 22
  , 0
  , 0
  , 0
  , 27
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 28
  , 0
  , 0
  , 0
  , 0
  , 47
  , 44
  , 44
  , 44
  , 44
  , 44
  , 44
  , 44
  , 44
  , 44
  , 44
  , 44
  , 44
  , 44
  , 44
  , 44
  , 44
  , 44
  , 44
  , 44
  , 44
  , 49
  , 5
  , 0
  , 0
  , 52
  , 52
  , 0
  , 52
  , 0
  , 13
  , 52
  , 0
  , 0
  , 52
  , 52
  , 0
  , 52
  , 3
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 38
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 0
  , 52
  , 0
  , 0
  , 52
  , 52
  , 0
  , 52
  , 3
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 34
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 0
  , 52
  , 0
  , 0
  , 52
  , 52
  , 0
  , 52
  , 3
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 0
  , 52
  , 0
  , 0
  , 52
  , 52
  , 0
  , 52
  , 3
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 0
  , 52
  , 0
  , 0
  , 52
  , 52
  , 0
  , 52
  , 3
  , 52
  , 43
  , 43
  , 43
  , 43
  , 43
  , 43
  , 43
  , 43
  , 43
  , 43
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 0
  , 52
  , 0
  , 0
  , 52
  , 52
  , 0
  , 52
  , 3
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 35
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 25
  , 52
  , 52
  , 0
  , 52
  , 0
  , 0
  , 52
  , 0
  , 0
  , 52
  , 52
  , 0
  , 52
  , 3
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 53
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 0
  , 52
  , 0
  , 0
  , 52
  , 52
  , 0
  , 52
  , 3
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 0
  , 52
  , 0
  , 0
  , 52
  , 52
  , 0
  , 52
  , 3
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 42
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 0
  , 52
  , 0
  , 0
  , 52
  , 52
  , 0
  , 52
  , 3
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 52
  , 52
  , 36
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 0
  , 52
  , 0
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 52
  , 0
  , 52
  , 55
  , 55
  , 0
  , 55
  , 0
  , 0
  , 55
  , 0
  , 0
  , 55
  , 55
  , 0
  , 55
  , 0
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 0
  , 55
  , 55
  , 55
  , 55
  , 55
  , 0
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 0
  , 55
  , 0
  , 55
  , 0
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 55
  , 0
  , 55
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  , 0
  ]

alex_check :: Array Int Int
alex_check = listArray (0 :: Int, 1430)
  [ -1
  , 9
  , 10
  , 11
  , 12
  , 13
  , 115
  , 101
  , 101
  , 105
  , 97
  , 99
  , 101
  , 105
  , 109
  , 101
  , 117
  , 98
  , 32
  , 102
  , 32
  , 100
  , 62
  , 114
  , 32
  , 33
  , 34
  , 46
  , 36
  , 37
  , 38
  , 39
  , 40
  , 41
  , 42
  , 43
  , 44
  , 45
  , 46
  , 47
  , 48
  , 49
  , 50
  , 51
  , 52
  , 53
  , 54
  , 55
  , 56
  , 57
  , 58
  , 59
  , 60
  , 61
  , 62
  , 63
  , 114
  , 65
  , 66
  , 67
  , 68
  , 69
  , 70
  , 71
  , 72
  , 73
  , 74
  , 75
  , 76
  , 77
  , 78
  , 79
  , 80
  , 81
  , 82
  , 83
  , 84
  , 85
  , 86
  , 87
  , 88
  , 89
  , 90
  , 91
  , 116
  , 93
  , 32
  , 95
  , 108
  , 97
  , 98
  , 99
  , 100
  , 101
  , 102
  , 103
  , 104
  , 105
  , 106
  , 107
  , 108
  , 109
  , 110
  , 111
  , 112
  , 113
  , 114
  , 115
  , 116
  , 117
  , 118
  , 119
  , 120
  , 121
  , 122
  , 123
  , 124
  , 125
  , 126
  , 33
  , 34
  , 46
  , 36
  , -1
  , 70
  , 39
  , -1
  , -1
  , 42
  , 43
  , 32
  , 45
  , 78
  , 47
  , 48
  , 49
  , 50
  , 51
  , 52
  , 53
  , 54
  , 55
  , 56
  , 57
  , 32
  , 59
  , 60
  , 61
  , 62
  , 63
  , -1
  , 65
  , 66
  , 67
  , 68
  , 69
  , 70
  , 71
  , 72
  , 73
  , 74
  , 75
  , 76
  , 77
  , 78
  , 79
  , 80
  , 81
  , 82
  , 83
  , 84
  , 85
  , 86
  , 87
  , 88
  , 89
  , 90
  , 91
  , -1
  , 93
  , -1
  , 95
  , -1
  , 97
  , 98
  , 99
  , 100
  , 101
  , 102
  , 103
  , 104
  , 105
  , 106
  , 107
  , 108
  , 109
  , 110
  , 111
  , 112
  , 113
  , 114
  , 115
  , 116
  , 117
  , 118
  , 119
  , 120
  , 121
  , 122
  , 123
  , 112
  , 125
  , 9
  , 10
  , 11
  , 12
  , 13
  , 9
  , 10
  , 11
  , 12
  , 13
  , 38
  , 111
  , 9
  , 10
  , 11
  , 12
  , 13
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , 32
  , 32
  , -1
  , -1
  , -1
  , 32
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , 32
  , -1
  , -1
  , -1
  , -1
  , 37
  , 48
  , 49
  , 50
  , 51
  , 52
  , 53
  , 54
  , 55
  , 56
  , 57
  , 48
  , 49
  , 50
  , 51
  , 52
  , 53
  , 54
  , 55
  , 56
  , 57
  , 58
  , 70
  , -1
  , -1
  , 33
  , 34
  , -1
  , 36
  , -1
  , 78
  , 39
  , -1
  , -1
  , 42
  , 43
  , -1
  , 45
  , 46
  , 47
  , 48
  , 49
  , 50
  , 51
  , 52
  , 53
  , 54
  , 55
  , 56
  , 57
  , -1
  , 59
  , 60
  , 61
  , 62
  , 63
  , 124
  , 65
  , 66
  , 67
  , 68
  , 69
  , 70
  , 71
  , 72
  , 73
  , 74
  , 75
  , 76
  , 77
  , 78
  , 79
  , 80
  , 81
  , 82
  , 83
  , 84
  , 85
  , 86
  , 87
  , 88
  , 89
  , 90
  , 91
  , -1
  , 93
  , -1
  , 95
  , -1
  , 97
  , 98
  , 99
  , 100
  , 101
  , 102
  , 103
  , 104
  , 105
  , 106
  , 107
  , 108
  , 109
  , 110
  , 111
  , 112
  , 113
  , 114
  , 115
  , 116
  , 117
  , 118
  , 119
  , 120
  , 121
  , 122
  , 123
  , -1
  , 125
  , 33
  , 34
  , -1
  , 36
  , -1
  , -1
  , 39
  , -1
  , -1
  , 42
  , 43
  , -1
  , 45
  , 46
  , 47
  , 48
  , 49
  , 50
  , 51
  , 52
  , 53
  , 54
  , 55
  , 56
  , 57
  , -1
  , 59
  , 60
  , 61
  , 62
  , 63
  , -1
  , 65
  , 66
  , 67
  , 68
  , 69
  , 70
  , 71
  , 72
  , 73
  , 74
  , 75
  , 76
  , 77
  , 78
  , 79
  , 80
  , 81
  , 82
  , 83
  , 84
  , 85
  , 86
  , 87
  , 88
  , 89
  , 90
  , 91
  , -1
  , 93
  , -1
  , 95
  , -1
  , 97
  , 98
  , 99
  , 100
  , 101
  , 102
  , 103
  , 104
  , 105
  , 106
  , 107
  , 108
  , 109
  , 110
  , 111
  , 112
  , 113
  , 114
  , 115
  , 116
  , 117
  , 118
  , 119
  , 120
  , 121
  , 122
  , 123
  , -1
  , 125
  , 33
  , 34
  , -1
  , 36
  , -1
  , -1
  , 39
  , -1
  , -1
  , 42
  , 43
  , -1
  , 45
  , 46
  , 47
  , 48
  , 49
  , 50
  , 51
  , 52
  , 53
  , 54
  , 55
  , 56
  , 57
  , -1
  , 59
  , 60
  , 61
  , 62
  , 63
  , -1
  , 65
  , 66
  , 67
  , 68
  , 69
  , 70
  , 71
  , 72
  , 73
  , 74
  , 75
  , 76
  , 77
  , 78
  , 79
  , 80
  , 81
  , 82
  , 83
  , 84
  , 85
  , 86
  , 87
  , 88
  , 89
  , 90
  , 91
  , -1
  , 93
  , -1
  , 95
  , -1
  , 97
  , 98
  , 99
  , 100
  , 101
  , 102
  , 103
  , 104
  , 105
  , 106
  , 107
  , 108
  , 109
  , 110
  , 111
  , 112
  , 113
  , 114
  , 115
  , 116
  , 117
  , 118
  , 119
  , 120
  , 121
  , 122
  , 123
  , -1
  , 125
  , 33
  , 34
  , -1
  , 36
  , -1
  , -1
  , 39
  , -1
  , -1
  , 42
  , 43
  , -1
  , 45
  , 46
  , 47
  , 48
  , 49
  , 50
  , 51
  , 52
  , 53
  , 54
  , 55
  , 56
  , 57
  , -1
  , 59
  , 60
  , 61
  , 62
  , 63
  , -1
  , 65
  , 66
  , 67
  , 68
  , 69
  , 70
  , 71
  , 72
  , 73
  , 74
  , 75
  , 76
  , 77
  , 78
  , 79
  , 80
  , 81
  , 82
  , 83
  , 84
  , 85
  , 86
  , 87
  , 88
  , 89
  , 90
  , 91
  , -1
  , 93
  , -1
  , 95
  , -1
  , 97
  , 98
  , 99
  , 100
  , 101
  , 102
  , 103
  , 104
  , 105
  , 106
  , 107
  , 108
  , 109
  , 110
  , 111
  , 112
  , 113
  , 114
  , 115
  , 116
  , 117
  , 118
  , 119
  , 120
  , 121
  , 122
  , 123
  , -1
  , 125
  , 33
  , 34
  , -1
  , 36
  , -1
  , -1
  , 39
  , -1
  , -1
  , 42
  , 43
  , -1
  , 45
  , 46
  , 47
  , 48
  , 49
  , 50
  , 51
  , 52
  , 53
  , 54
  , 55
  , 56
  , 57
  , -1
  , 59
  , 60
  , 61
  , 62
  , 63
  , -1
  , 65
  , 66
  , 67
  , 68
  , 69
  , 70
  , 71
  , 72
  , 73
  , 74
  , 75
  , 76
  , 77
  , 78
  , 79
  , 80
  , 81
  , 82
  , 83
  , 84
  , 85
  , 86
  , 87
  , 88
  , 89
  , 90
  , 91
  , -1
  , 93
  , -1
  , 95
  , -1
  , 97
  , 98
  , 99
  , 100
  , 101
  , 102
  , 103
  , 104
  , 105
  , 106
  , 107
  , 108
  , 109
  , 110
  , 111
  , 112
  , 113
  , 114
  , 115
  , 116
  , 117
  , 118
  , 119
  , 120
  , 121
  , 122
  , 123
  , -1
  , 125
  , 33
  , 34
  , -1
  , 36
  , -1
  , -1
  , 39
  , -1
  , -1
  , 42
  , 43
  , -1
  , 45
  , 46
  , 47
  , 48
  , 49
  , 50
  , 51
  , 52
  , 53
  , 54
  , 55
  , 56
  , 57
  , -1
  , 59
  , 60
  , 61
  , 62
  , 63
  , -1
  , 65
  , 66
  , 67
  , 68
  , 69
  , 70
  , 71
  , 72
  , 73
  , 74
  , 75
  , 76
  , 77
  , 78
  , 79
  , 80
  , 81
  , 82
  , 83
  , 84
  , 85
  , 86
  , 87
  , 88
  , 89
  , 90
  , 91
  , -1
  , 93
  , -1
  , 95
  , -1
  , 97
  , 98
  , 99
  , 100
  , 101
  , 102
  , 103
  , 104
  , 105
  , 106
  , 107
  , 108
  , 109
  , 110
  , 111
  , 112
  , 113
  , 114
  , 115
  , 116
  , 117
  , 118
  , 119
  , 120
  , 121
  , 122
  , 123
  , -1
  , 125
  , 126
  , 33
  , 34
  , -1
  , 36
  , -1
  , -1
  , 39
  , -1
  , -1
  , 42
  , 43
  , -1
  , 45
  , 46
  , 47
  , 48
  , 49
  , 50
  , 51
  , 52
  , 53
  , 54
  , 55
  , 56
  , 57
  , -1
  , 59
  , 60
  , 61
  , 62
  , 63
  , -1
  , 65
  , 66
  , 67
  , 68
  , 69
  , 70
  , 71
  , 72
  , 73
  , 74
  , 75
  , 76
  , 77
  , 78
  , 79
  , 80
  , 81
  , 82
  , 83
  , 84
  , 85
  , 86
  , 87
  , 88
  , 89
  , 90
  , 91
  , -1
  , 93
  , -1
  , 95
  , -1
  , 97
  , 98
  , 99
  , 100
  , 101
  , 102
  , 103
  , 104
  , 105
  , 106
  , 107
  , 108
  , 109
  , 110
  , 111
  , 112
  , 113
  , 114
  , 115
  , 116
  , 117
  , 118
  , 119
  , 120
  , 121
  , 122
  , 123
  , -1
  , 125
  , 33
  , 34
  , -1
  , 36
  , -1
  , -1
  , 39
  , -1
  , -1
  , 42
  , 43
  , -1
  , 45
  , 46
  , 47
  , 48
  , 49
  , 50
  , 51
  , 52
  , 53
  , 54
  , 55
  , 56
  , 57
  , -1
  , 59
  , 60
  , 61
  , 62
  , 63
  , -1
  , 65
  , 66
  , 67
  , 68
  , 69
  , 70
  , 71
  , 72
  , 73
  , 74
  , 75
  , 76
  , 77
  , 78
  , 79
  , 80
  , 81
  , 82
  , 83
  , 84
  , 85
  , 86
  , 87
  , 88
  , 89
  , 90
  , 91
  , -1
  , 93
  , -1
  , 95
  , -1
  , 97
  , 98
  , 99
  , 100
  , 101
  , 102
  , 103
  , 104
  , 105
  , 106
  , 107
  , 108
  , 109
  , 110
  , 111
  , 112
  , 113
  , 114
  , 115
  , 116
  , 117
  , 118
  , 119
  , 120
  , 121
  , 122
  , 123
  , -1
  , 125
  , 33
  , 34
  , -1
  , 36
  , -1
  , -1
  , 39
  , -1
  , -1
  , 42
  , 43
  , -1
  , 45
  , 46
  , 47
  , 48
  , 49
  , 50
  , 51
  , 52
  , 53
  , 54
  , 55
  , 56
  , 57
  , -1
  , 59
  , 60
  , 61
  , 62
  , 63
  , -1
  , 65
  , 66
  , 67
  , 68
  , 69
  , 70
  , 71
  , 72
  , 73
  , 74
  , 75
  , 76
  , 77
  , 78
  , 79
  , 80
  , 81
  , 82
  , 83
  , 84
  , 85
  , 86
  , 87
  , 88
  , 89
  , 90
  , 91
  , -1
  , 93
  , -1
  , 95
  , -1
  , 97
  , 98
  , 99
  , 100
  , 101
  , 102
  , 103
  , 104
  , 105
  , 106
  , 107
  , 108
  , 109
  , 110
  , 111
  , 112
  , 113
  , 114
  , 115
  , 116
  , 117
  , 118
  , 119
  , 120
  , 121
  , 122
  , 123
  , -1
  , 125
  , 33
  , 34
  , -1
  , 36
  , -1
  , -1
  , 39
  , -1
  , -1
  , 42
  , 43
  , -1
  , 45
  , 46
  , 47
  , 48
  , 49
  , 50
  , 51
  , 52
  , 53
  , 54
  , 55
  , 56
  , 57
  , -1
  , 59
  , 60
  , 61
  , 62
  , 63
  , -1
  , 65
  , 66
  , 67
  , 68
  , 69
  , 70
  , 71
  , 72
  , 73
  , 74
  , 75
  , 76
  , 77
  , 78
  , 79
  , 80
  , 81
  , 82
  , 83
  , 84
  , 85
  , 86
  , 87
  , 88
  , 89
  , 90
  , 91
  , -1
  , 93
  , -1
  , 95
  , -1
  , 97
  , 98
  , 99
  , 100
  , 101
  , 102
  , 103
  , 104
  , 105
  , 106
  , 107
  , 108
  , 109
  , 110
  , 111
  , 112
  , 113
  , 114
  , 115
  , 116
  , 117
  , 118
  , 119
  , 120
  , 121
  , 122
  , 123
  , -1
  , 125
  , 33
  , 34
  , -1
  , 36
  , -1
  , -1
  , 39
  , -1
  , -1
  , 42
  , 43
  , -1
  , 45
  , -1
  , 47
  , 48
  , 49
  , 50
  , 51
  , 52
  , 53
  , 54
  , 55
  , 56
  , 57
  , -1
  , 59
  , 60
  , 61
  , 62
  , 63
  , -1
  , 65
  , 66
  , 67
  , 68
  , 69
  , 70
  , 71
  , 72
  , 73
  , 74
  , 75
  , 76
  , 77
  , 78
  , 79
  , 80
  , 81
  , 82
  , 83
  , 84
  , 85
  , 86
  , 87
  , 88
  , 89
  , 90
  , 91
  , -1
  , 93
  , -1
  , 95
  , -1
  , 97
  , 98
  , 99
  , 100
  , 101
  , 102
  , 103
  , 104
  , 105
  , 106
  , 107
  , 108
  , 109
  , 110
  , 111
  , 112
  , 113
  , 114
  , 115
  , 116
  , 117
  , 118
  , 119
  , 120
  , 121
  , 122
  , 123
  , -1
  , 125
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  ]

alex_deflt :: Array Int Int
alex_deflt = listArray (0 :: Int, 57)
  [ -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  , -1
  ]

alex_accept = listArray (0 :: Int, 57)
  [ AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccNone
  , AlexAccSkip
  , AlexAccSkip
  , AlexAccSkip
  , AlexAcc 28
  , AlexAcc 27
  , AlexAcc 26
  , AlexAcc 25
  , AlexAcc 24
  , AlexAcc 23
  , AlexAcc 22
  , AlexAcc 21
  , AlexAcc 20
  , AlexAcc 19
  , AlexAcc 18
  , AlexAcc 17
  , AlexAcc 16
  , AlexAcc 15
  , AlexAcc 14
  , AlexAcc 13
  , AlexAcc 12
  , AlexAcc 11
  , AlexAcc 10
  , AlexAcc 9
  , AlexAcc 8
  , AlexAcc 7
  , AlexAcc 6
  , AlexAcc 5
  , AlexAcc 4
  , AlexAcc 3
  , AlexAcc 2
  , AlexAcc 1
  , AlexAcc 0
  ]

alex_actions = array (0 :: Int, 29)
  [ (28,alex_action_2)
  , (27,alex_action_3)
  , (26,alex_action_4)
  , (25,alex_action_5)
  , (24,alex_action_6)
  , (23,alex_action_8)
  , (22,alex_action_9)
  , (21,alex_action_10)
  , (20,alex_action_11)
  , (19,alex_action_12)
  , (18,alex_action_13)
  , (17,alex_action_14)
  , (16,alex_action_15)
  , (15,alex_action_16)
  , (14,alex_action_17)
  , (13,alex_action_17)
  , (12,alex_action_18)
  , (11,alex_action_19)
  , (10,alex_action_20)
  , (9,alex_action_20)
  , (8,alex_action_21)
  , (7,alex_action_22)
  , (6,alex_action_22)
  , (5,alex_action_22)
  , (4,alex_action_22)
  , (3,alex_action_22)
  , (2,alex_action_23)
  , (1,alex_action_26)
  , (0,alex_action_27)
  ]

{-# LINE 78 "DTS/Alligator/AlexHappy/Lexer.x" #-}


data Token
  = TokenNum Int
  | TokenPreNum
  | TokenEOF
  | TokenFOF
  | TokenHead
  | TokenCoron
  | TokenComma
  | TokenConne String
  | TokenWord String
  | TokenFile
  | TokenAnd
  | TokenRBracket
  | TokenLBracket
  | TokenPeriod
  deriving (Eq,Show)

scanTokens = alexScanTokens


alex_action_2 =  \s -> TokenAnd 
alex_action_3 =  \s -> TokenRBracket 
alex_action_4 =  \s -> TokenLBracket 
alex_action_5 =  \s -> TokenConne s
alex_action_6 =  \s -> TokenConne s
alex_action_7 =  \s -> TokenConne s
alex_action_8 =  \s -> TokenConne s
alex_action_9 =  \s -> TokenConne s
alex_action_10 =  \s -> TokenConne s
alex_action_11 =  \s -> TokenConne s
alex_action_12 =  \s -> TokenConne s
alex_action_13 =  \s -> TokenConne s
alex_action_14 =  \s -> TokenComma 
alex_action_15 =  \s -> TokenPeriod 
alex_action_16 =  \s -> TokenFOF 
alex_action_17 =  \s -> TokenNum (read s) 
alex_action_18 =  \s -> TokenPreNum
alex_action_19 =  \s -> TokenFile 
alex_action_20 =  \s -> TokenHead 
alex_action_21 =  \s -> TokenCoron 
alex_action_22 =  \s -> TokenWord s 
alex_action_23 =  \s -> TokenWord s 
alex_action_24 =  \s -> TokenWord "<->"
alex_action_25 =  \s -> TokenWord "->"
alex_action_26 =  \s -> TokenWord "..."
alex_action_27 =  \s -> TokenWord "..."
{-# LINE 1 "templates/GenericTemplate.hs" #-}
-- -----------------------------------------------------------------------------
-- ALEX TEMPLATE
--
-- This code is in the PUBLIC DOMAIN; you may copy it freely and use
-- it for any purpose whatsoever.

-- -----------------------------------------------------------------------------
-- INTERNALS and main scanner engine































































alexIndexInt16OffAddr arr off = arr ! off




















alexIndexInt32OffAddr arr off = arr ! off











quickIndex arr i = arr ! i


-- -----------------------------------------------------------------------------
-- Main lexing routines

data AlexReturn a
  = AlexEOF
  | AlexError  !AlexInput
  | AlexSkip   !AlexInput !Int
  | AlexToken  !AlexInput !Int a

-- alexScan :: AlexInput -> StartCode -> AlexReturn a
alexScan input__ (sc)
  = alexScanUser undefined input__ (sc)

alexScanUser user__ input__ (sc)
  = case alex_scan_tkn user__ input__ (0) input__ sc AlexNone of
  (AlexNone, input__') ->
    case alexGetByte input__ of
      Nothing ->



                                   AlexEOF
      Just _ ->



                                   AlexError input__'

  (AlexLastSkip input__'' len, _) ->



    AlexSkip input__'' len

  (AlexLastAcc k input__''' len, _) ->



    AlexToken input__''' len (alex_actions ! k)


-- Push the input through the DFA, remembering the most recent accepting
-- state it encountered.

alex_scan_tkn user__ orig_input len input__ s last_acc =
  input__ `seq` -- strict in the input
  let
  new_acc = (check_accs (alex_accept `quickIndex` (s)))
  in
  new_acc `seq`
  case alexGetByte input__ of
     Nothing -> (new_acc, input__)
     Just (c, new_input) ->



      case fromIntegral c of { (ord_c) ->
        let
                base   = alexIndexInt32OffAddr alex_base s
                offset = (base + ord_c)
                check  = alexIndexInt16OffAddr alex_check offset

                new_s = if (offset >= (0)) && (check == ord_c)
                          then alexIndexInt16OffAddr alex_table offset
                          else alexIndexInt16OffAddr alex_deflt s
        in
        case new_s of
            (-1) -> (new_acc, input__)
                -- on an error, we want to keep the input *before* the
                -- character that failed, not after.
            _ -> alex_scan_tkn user__ orig_input (if c < 0x80 || c >= 0xC0 then (len + (1)) else len)
                                                -- note that the length is increased ONLY if this is the 1st byte in a char encoding)
                        new_input new_s new_acc
      }
  where
        check_accs (AlexAccNone) = last_acc
        check_accs (AlexAcc a  ) = AlexLastAcc a input__ (len)
        check_accs (AlexAccSkip) = AlexLastSkip  input__ (len)

        check_accs (AlexAccPred a predx rest)
           | predx user__ orig_input (len) input__
           = AlexLastAcc a input__ (len)
           | otherwise
           = check_accs rest
        check_accs (AlexAccSkipPred predx rest)
           | predx user__ orig_input (len) input__
           = AlexLastSkip input__ (len)
           | otherwise
           = check_accs rest


data AlexLastAcc
  = AlexNone
  | AlexLastAcc !Int !AlexInput !Int
  | AlexLastSkip     !AlexInput !Int

data AlexAcc user
  = AlexAccNone
  | AlexAcc Int
  | AlexAccSkip

  | AlexAccPred Int (AlexAccPred user) (AlexAcc user)
  | AlexAccSkipPred (AlexAccPred user) (AlexAcc user)

type AlexAccPred user = user -> AlexInput -> Int -> AlexInput -> Bool

-- -----------------------------------------------------------------------------
-- Predicates on a rule

alexAndPred p1 p2 user__ in1 len in2
  = p1 user__ in1 len in2 && p2 user__ in1 len in2

--alexPrevCharIsPred :: Char -> AlexAccPred _
alexPrevCharIs c _ input__ _ _ = c == alexInputPrevChar input__

alexPrevCharMatches f _ input__ _ _ = f (alexInputPrevChar input__)

--alexPrevCharIsOneOfPred :: Array Char Bool -> AlexAccPred _
alexPrevCharIsOneOf arr _ input__ _ _ = arr ! alexInputPrevChar input__

--alexRightContext :: Int -> AlexAccPred _
alexRightContext (sc) user__ _ _ input__ =
     case alex_scan_tkn user__ input__ (0) input__ sc AlexNone of
          (AlexNone, _) -> False
          _ -> True
        -- TODO: there's no need to find the longest
        -- match when checking the right context, just
        -- the first match will do.
