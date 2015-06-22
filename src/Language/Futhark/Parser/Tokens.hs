-- | Lexical tokens generated by the lexer and consumed by the parser.
--
-- Probably the most boring module in the compiler.
module Language.Futhark.Parser.Tokens
  ( Token(..)
  )
  where

import Language.Futhark.Core

-- | A lexical token.  It does not itself contain position
-- information, so in practice the parser will consume tokens tagged
-- with a source position.
data Token = IF
           | THEN
           | ELSE
           | LET
           | LOOP
           | IN
           | INT
           | BOOL
           | CERT
           | CHAR
           | REAL
           | ID { idName :: Name }
           | STRINGLIT { stringLit :: String }
           | INTLIT { intLit :: Int32 }
           | REALLIT { realLit :: Double }
           | CHARLIT { charLit :: Char }
           | PLUS
           | MINUS
           | TIMES
           | DIVIDE
           | MOD
           | EQU
           | EQU2
           | LTH
           | GTH
           | LEQ
           | GEQ
           | POW
           | SHIFTL
           | SHIFTR
           | BOR
           | BAND
           | XOR
           | LPAR
           | RPAR
           | LBRACKET
           | RBRACKET
           | LCURLY
           | RCURLY
           | COMMA
           | UNDERSCORE
           | FUN
           | FN
           | ARROW
           | SETTO
           | FOR
           | DO
           | WITH
           | SIZE
           | IOTA
           | REPLICATE
           | MAP
           | CONCATMAP
           | REDUCE
           | RESHAPE
           | REARRANGE
           | TRANSPOSE
           | ZIPWITH
           | ZIP
           | UNZIP
           | SCAN
           | SPLIT
           | CONCAT
           | FILTER
           | PARTITION
           | REDOMAP
           | TRUE
           | FALSE
           | CHECKED
           | TILDE
           | AND
           | OR
           | OP
           | EMPTY
           | COPY
           | ASSERT
           | WHILE
           | STREAM_MAP
           | STREAM_MAPMAX
           | STREAM_MAPPER
           | STREAM_MAPPERMAX
           | STREAM_RED
           | STREAM_REDMAX
           | STREAM_REDPER
           | STREAM_REDPERMAX
           | STREAM_SEQ
           | STREAM_SEQMAX
           | BANG
           | ABS
           | SIGNUM
           | EOF
             deriving (Show, Eq)
