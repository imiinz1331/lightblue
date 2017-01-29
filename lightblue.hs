{-# OPTIONS -Wall #-}
{-# LANGUAGE OverloadedStrings #-}

import Options.Applicative                         -- optparse-applicative
import Options.Applicative.Help.Core (parserUsage) -- optparse-applicative
import Data.Semigroup ((<>))              -- semigroup
import qualified Data.Text.Lazy as T      --text
import qualified Data.Text.Lazy.IO as T   --text
import qualified Data.List as L           --base
import qualified Data.Map as M            --
import qualified Data.Time as Time        --time
import qualified Data.Ratio as R
import qualified Data.Fixed as F
import qualified System.IO as S           --base
import qualified System.Environment as E -- base
import qualified Parser.ChartParser as CP
import qualified Parser.Japanese.MyLexicon as LEX
import qualified DTS.UDTT as DTS
import qualified Interface as I
import qualified Interface.Text as T
import qualified Interface.HTML as HTML
import qualified Interface.JSeM as J
import qualified DTS.Prover.TypeChecker as Ty
import qualified DTS.Prover.Judgement as Ty

data Options =  
  Version 
  | Stat 
  | JSEM FilePath
  | Corpus FilePath
  | Debug Int Int
  | Options
  { task :: String
  , format :: String
  , nBest  :: Int
  , showTypeCheck :: Bool
  , showExecutionTime :: Bool
  } deriving (Show, Eq)

main :: IO()
main = execParser opts >>= lightblueMain 
  where opts = info (helper <*> optionParser)
                 ( fullDesc
                 <> progDesc "echo <sentence> | ./lightblue\n echo <sentence> | ./lightblue"
                 <> header "lightblue - a Japanese CCG parser with DTS representations (c) Bekki Laboratory" )

{-
  <$> :: (a -> b) -> Parser a -> Parser b
  <*> :: Parser (a -> b) -> Parser a -> Parser b
-}

optionParser :: Parser Options
optionParser = 
  flag' Version ( long "version" 
                <> short 'v' 
                <> help "Show the version of lightblue parser" )
  <|> 
  flag' Stat ( long "stat" 
             <> help "Show the statistics of ligthblue parser" )
  <|>
  JSEM 
    <$> strOption
        ( long "jsem"
        <> metavar "FILEPATH"
        <> help "Parse JSeM data" )
  <|>
  Corpus 
    <$> strOption
        ( long "corpus"
        <> metavar "FILEPATH"
        <> help "Parse BCCWJ corpus" )
  <|>
  subparser (command "debug" (info (Debug
                                     <$> argument auto idm
                                       --( help "from" <> showDefault <> value 0 <> metavar "N" )
                                     <*> argument auto 
                                       ( help "to" <> showDefault <> value 1 <> metavar "N" ))
                                   (progDesc "'lightblue debug i j' shows all the parsing results between the pivots i and j." )))
  <|>
  Options 
    <$> strOption
      ( long "task"
      <> short 't'
      <> metavar "parse|infer|postag|numeration"
      <> help ("Execute the specified task"
                --"Usage for parse: cat <sentence> | lightblue -t parse > output.html "
                --"Usage for infer: cat <textfile> | lightblue -t infer > output.html "
                --"where <textfile> consists of premises and a coclusion" 
                --"(with one sentence per each line)"
              )
      <> showDefault
      <> value "parse" )
    <*> strOption 
      ( long "output"
      <> short 'o'
      <> metavar "text|tex|xml|html"
      <> help "Print result in the specified format" 
      <> showDefault
      <> value "html" )
    <*> option auto 
      ( long "nbest"
      <> short 'n'
      <> help "Show N-best derivations"
      <> showDefault
      <> value 1
      <> metavar "N" )
    <*> switch 
      ( long "typecheck"
      <> help "Execute typechecking for the SR" )
    <*> switch 
      ( long "time"
      <> help "Show the execution time in stderr" )

lightblueMain :: Options -> IO()
lightblueMain Version = showVersion
lightblueMain Stat = showStat
lightblueMain (JSEM filepath) = parseJSeM filepath
lightblueMain (Corpus filepath) = parseCorpus filepath
lightblueMain (Debug i j) =  do
  sentence <- T.getLine
  chart <- CP.parse 24 sentence
  I.printNodesInHTML S.stdout 100 False $ L.concat $ map (\(_,nodes) -> nodes) $ filter (\((x,y),_) -> i <= x && y <= j) $ M.toList chart
lightblueMain options = do
  start    <- Time.getCurrentTime
  sentence <- T.getLine
  nodes    <- CP.simpleParse 24 sentence
  stop     <- Time.getCurrentTime
  let time = Time.diffUTCTime stop start
  case (task options, format options) of
    ("infer",_) -> checkEntailment
    ("postag",_) -> I.posTagger S.stdout nodes
    ("numeration",_) -> I.printNumeration S.stdout sentence
    ("parse","html") -> I.printNodesInHTML S.stdout (nBest options) (showTypeCheck options) nodes
    ("parse","text") -> I.printNodesInText S.stdout (nBest options) (showTypeCheck options) nodes
    ("parse","tex")  -> I.printNodesInTeX  S.stdout (nBest options) (showTypeCheck options) nodes
    ("parse","xml")  -> I.printNodesInXML  S.stdout sentence (nBest options) nodes
    (t,f) -> S.hPutStrLn S.stderr $ show $ parserUsage defaultPrefs optionParser $ "task=" ++ t ++ ", format=" ++ f ++ ": Not supported."
  if showExecutionTime options 
     then S.hPutStrLn S.stderr $ "Total Execution Time: " ++ show time
     else return ()

-- | lightblue --version
-- |
showVersion :: IO()
showVersion = do
  T.putStr "lightblue version: "
  lightbluepath <- E.getEnv "LIGHTBLUE"
  cabal <- T.readFile $ lightbluepath ++ "lightblue.cabal"
  T.putStrLn $ last $ T.words $ head $ filter (T.isPrefixOf "version:") $ T.lines cabal

-- | lightblue --status
-- |
showStat :: IO()
showStat = do
  putStrLn "lightblue: "
  putStr "  "
  putStr $ show $ length $ LEX.emptyCategories
  putStrLn " empty categories from CCG book"
  putStr "  "
  putStr $ show $ length $ LEX.myLexicon
  putStrLn " lexical entries for closed words from CCG book"
  jumandicpath <- E.getEnv "LIGHTBLUE"
  jumandic <- T.readFile $ jumandicpath ++ "Parser/Japanese/Juman.dic"
  putStr "  "
  putStr $ show $ length $ T.lines jumandic
  putStrLn " lexical entries for open words from JUMAN++"

{-
-- | lightblue --fuman (hidden option)
-- | transforms an input (from stdin) each of whose line is a json entry
-- | into an output (to stdout) each of whose line is a paragraph.
-- | Usage:
-- | cat <file> | lightblue --fuman | head -n | Fuman/para2sentence > ...txt
-- |
fuman2text :: IO()
fuman2text = do
  jsonStrings <- T.getContents
  mapM_ T.putStrLn $ M.catMaybes $ map (\j -> j ^? key "fuman" . _String) $ T.lines jsonStrings
-}

-- | $ ligthblue --jsem ../JSeM_beta/JSeM_beta_150415.xml
-- | 
parseJSeM :: FilePath -> IO()
parseJSeM filepath = do
  content <- T.readFile filepath
  mapM_ processJSeMData $ J.parseJSeM content

processJSeMData :: J.JSeMData -> IO()
processJSeMData jsemdata = do
  T.putStrLn $ T.concat ["id [", J.jsem_id (jsemdata), "]"]
  mapM_ (\p -> do {T.putStr $ T.concat ["P: ", p, "\n"]}) $ J.premise jsemdata
  T.putStr $ T.concat ["H: ", J.hypothesis jsemdata, "\n"]
  psems <- mapM parseText $ J.premise jsemdata
  hsem <- parseText $ J.hypothesis jsemdata
  let sem = DTS.betaReduce $ currying psems hsem
  T.putStrLn $ T.toText sem

currying :: [DTS.Preterm] -> DTS.Preterm -> DTS.Preterm
currying [] preterm = preterm
currying (p:ps) preterm = DTS.Pi p (currying ps preterm)

parseText :: T.Text -> IO(DTS.Preterm)
parseText sentence = do
  nodes <- CP.simpleParse 16 sentence
  return $ CP.sem (head nodes)

{-
callCoq :: T.Text -> IO()
callCoq _ = do
  let coqcommand = T.concat ["echo -e \"Extraction Language Scheme.\nParameter A:Prop.\nParameter B:Prop.\nTheorem id: A -> B -> A.\nExtraction id.\n\" | coqtop 2> /dev/null | awk '{if($0 != \"\") {print $0}}' | tail -n 2"]
  (_, stdout, _, _) <- S.runInteractiveCommand $ T.unpack coqcommand
  t <- T.hGetContents stdout
  T.putStrLn $ T.replace "\n" "" $ T.strip t
-}

-- | lightblue --corpus filepath
-- |
parseCorpus :: FilePath -> IO()
parseCorpus filepath = do
    start <- Time.getCurrentTime
    sentences <- T.readFile filepath
    (i,j,k,total) <- L.foldl' parseSentence (return (0,0,0,0)) $ filter isSentence $ T.lines sentences
    stop <- Time.getCurrentTime
    let totaltime = Time.diffUTCTime stop start
    S.hPutStrLn S.stdout $ "Results: Full:Partial:Error = " 
                           ++(show i)++":"++(show j)++":"++(show k)
                           ++ ", Full/Total = "
                           ++(show i)++"/"++(show total)
                           ++" (" 
                           ++ (show $ ((fromRational ((toEnum i R.% toEnum total)*100))::F.Fixed F.E3)) 
                           ++ "%)"
    S.hPutStrLn S.stdout $ "Execution Time: " 
                           ++ show totaltime 
                           ++ " (average: " 
                           ++ (show $ ((fromRational ((toEnum (fromEnum totaltime)) R.% toEnum (j*1000000000000)))::F.Fixed F.E3)) 
                           ++ "s/sentence)"
    where isSentence t = not (t == T.empty || "（" `T.isSuffixOf` t)

parseSentence :: IO(Int,Int,Int,Int) -- ^ (The number of fully succeeded, partially succeeded, failed, and total parses)
                 -> T.Text           -- ^ A next sentence to parse
                 -> IO(Int,Int,Int,Int)
parseSentence score sentence = do
  (i,j,k,total) <- score
  S.putStr $ "[" ++ show (total+1) ++ "] "
  T.putStrLn sentence
  let beam = 24
  chart <- CP.parse beam sentence
  case CP.extractParseResult beam chart of
    CP.Full nodes -> 
       do
       T.putStr $ T.concat ["Fully parsed, Full:Partial:Failed = ", T.pack (show $ i+1), ":", T.pack (show j), ":", T.pack (show k), ", Full/Total = ", T.pack (show $ i+1), "/", T.pack (show $ total+1), " ("] 
       S.putStrLn $ percent (i+1,total+1) ++ "%)\n"
       T.putStrLn $ T.toText $ head $ nodes
       return (i+1,j,k,total+1)
    CP.Partial nodes -> 
       do
       T.putStr $ T.concat ["Partially parsed, Full:Partial:Failed = ", T.pack (show i), ":", T.pack (show $ j+1), ":", T.pack (show k), ", Full/Total = ", T.pack (show $ j+1), "/", T.pack (show $ total+1), " ("]
       S.putStrLn $ percent (i,total+1) ++ "%)\n"
       T.putStrLn $ T.toText $ head $ nodes
       return (i,j+1,k,total+1)
    CP.Failed ->
       do
       T.putStr $ T.concat ["Failed, Full:Partial:Failed = ", T.pack (show i), ":", T.pack (show $ j), ":", T.pack (show $ k+1), ", Full/Total = ", T.pack (show $ i+1), "/", T.pack (show $ total+1), " ("]
       S.putStrLn $ percent (i,total+1) ++ "%)\n"
       return (i,j,k+1,total+1)

percent :: (Int,Int) -> String
percent (i,j) = if j == 0
                   then show (0::F.Fixed F.E2)
                   else show ((fromRational (toEnum i R.% toEnum j)::F.Fixed F.E2) * 100)

-- | lightblue --ent
-- |
checkEntailment :: IO()
checkEntailment = do
  sentences <- T.getContents
  nodes <- mapM ((fmap head) . (CP.simpleParse 24)) (T.lines sentences)
  let premises = map (DTS.betaReduce . DTS.sigmaElimination . CP.sem) $ L.init $ nodes;
      conclusion = (DTS.betaReduce . DTS.sigmaElimination . CP.sem) $ L.last $ nodes;
      siglists = map CP.sig nodes;
  T.putStrLn HTML.htmlHeader4MathML
  mapM_ (\node -> mapM_ T.putStrLn [
                    HTML.startMathML, 
                    HTML.toMathML node, 
                    HTML.endMathML, 
                    "<hr size='15' />"
                    ]) nodes
  let proofdiagrams = Ty.proofSearch (reverse premises) (L.concat $ [("evt",DTS.Type),("entity",DTS.Type)]:siglists) conclusion
  if proofdiagrams == []
     then mapM_ T.putStrLn [
           "No proof diagrams for: ", 
           HTML.startMathML, 
           DTS.printProofSearchQuery (reverse premises) conclusion, 
           HTML.endMathML
           ]
      else do
           T.putStrLn HTML.startMathML
           mapM_ (T.putStrLn . Ty.utreeToMathML) proofdiagrams
           T.putStrLn HTML.endMathML
  T.putStrLn HTML.htmlFooter4MathML

