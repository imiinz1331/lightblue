module DTS.NeuralDTS.WordNet.WordNet (
  openDatabase
  , closeDatabase
  , getSynonyms
  , testDatabase
  ) where

import Control.Exception
import Database.SQLite.Simple
import Database.SQLite.Simple.FromRow
import qualified Data.Text.Lazy as T
import qualified Data.Text.Lazy.IO as TIO

wordnetDir = "src/DTS/NeuralDTS/WordNet/"

-- データベースを開く関数
openDatabase :: IO Connection
openDatabase = open (wordnetDir ++ "wnjpn.db")

-- データベースを閉じる関数
closeDatabase :: Connection -> IO ()
closeDatabase = close

-- 同義語を取得する関数
getSynonyms :: Connection -> T.Text -> IO [T.Text]
getSynonyms conn word = do
  let baseWord = T.takeWhile (/= '/') word
  rows <- query conn
    "SELECT w2.lemma \
    \FROM sense AS s1 \
    \JOIN word AS w1 ON s1.wordid = w1.wordid \
    \JOIN sense AS s2 ON s1.synset = s2.synset \
    \JOIN word AS w2 ON s2.wordid = w2.wordid \
    \WHERE w1.lemma = ? AND w1.lang = 'jpn' AND w2.lang = 'jpn' AND w2.lemma <> ?;"
    (baseWord, baseWord) :: IO [Only T.Text]
  return $ map fromOnly rows

-- SQLite を使った例
testDatabase :: IO ()
testDatabase = do
  result <- try (open (wordnetDir ++ "wnjpn.db")) :: IO (Either SomeException Connection)
  case result of
    Left ex -> putStrLn $ "Error opening database: " ++ show ex
    Right conn -> do
      putStrLn "Database opened successfully."
      -- `getSynonyms` を試す
      synonyms <- getSynonyms conn "ロシア"
      TIO.putStrLn $ "Synonyms for 'ロシア': " <> T.intercalate ", " synonyms
      close conn