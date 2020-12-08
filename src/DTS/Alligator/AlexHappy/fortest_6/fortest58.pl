:- style_check(-singleton).

:-use_module(swi_alligator,[prove/2]).
:- op(500,xfx,:).
:- op(300,yfx,-).
:- op(400,xfy,=>).
:- op(400,xfy,&).
:- op(300,fy,~).
:- dynamic checktheorem/6.
writeResult(Logic,ID,Prediction,Gold,Fname) :-
format('~w&~w&~w&~w&~w~n', [Logic,ID,Prediction,Gold,Fname]).
evalyes(Logic,ID) :-
  checkTheorem(Logic,ID,Context,Theorem,Gold,Fname),
  ( prove(Context,_ : Theorem) -> writeResult(Logic,ID,yes,Gold,Fname)).
evalno(Logic,ID) :-
  checkTheorem(Logic,ID,Context,Theorem,Gold,Fname),
  ( prove(Context,_ : ( ~ (Theorem) ) ) -> writeResult(Logic,ID,no,Gold,Fname) ).
checkTheorem(syn , 342 , [ type:set ,g2 : (( ( type ) ) -> ( ( prop ) )) , f : (( ( type ) ) -> ( ( prop ) )) , g1 : (( ( type ) ) -> ( ( prop ) )) , bigX : (( ( type ) ) -> ( ( prop ) )) ] , ( ~( ( ~( ~( ( ( ( ( f ) )-( ( ( g1 ) )-( ( bigX ) ) ) ) -> ( ( bigX ) ) ) & ( ( ( bigX ) ) -> ( ( ( f ) )-( ( ( g1 ) )-( ( bigX ) ) ) ) ) ) ) ) & ( ( ( ( ( f ) )-( ( ( g2 ) )-( ( bigX ) ) ) ) -> ( ( bigX ) ) ) & ( ( ( bigX ) ) -> ( ( ( f ) )-( ( ( g2 ) )-( ( bigX ) ) ) ) ) ) ) ) & ( ~( ( ( ( ( g1 ) )-( ( bigX ) ) ) -> ( ( ( g2 ) )-( ( bigX ) ) ) ) & ( ( ( ( g2 ) )-( ( bigX ) ) ) -> ( ( ( g1 ) )-( ( bigX ) ) ) ) ) ) , unknown , '../../TPTP-v7.3.0/Problems/SYN/SYN305-1.p').
checkTheorem(syn , 343 , [ type:set ,f : (( ( type ) ) -> ( ( prop ) )) , b : (( type )) , a : (( ( type ) ) -> ( ( prop ) )) , r : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigY : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , q : (( ( type ) ) -> ( ( prop ) )) , g : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p : (( ( type ) ) -> ( ( prop ) )) , bigX : (( ( type ) ) -> ( ( prop ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( ( p ) )-( ( bigX ) ) ) \/ ( ( ( q ) )-( ( ( g ) )-( ( bigX ) )-( ( bigX ) ) ) ) ) ) ) & ( ( ~( ( ( q ) )-( ( bigY ) ) ) ) \/ ( ( ( r ) )-( ( bigX ) )-( ( bigY ) ) ) ) ) ) & ( ( ~( ( ( r ) )-( ( a ) )-( ( a ) ) ) ) \/ ( ~( ( ( r ) )-( ( ( f ) )-( ( b ) ) )-( ( a ) ) ) ) ) ) ) & ( ( ( r ) )-( ( ( f ) )-( ( bigX ) ) )-( ( bigY ) ) ) ) ) & ( ( ( ( p ) )-( ( bigX ) ) ) \/ ( ~( ( ( p ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) ) ) ) & ( ~( ( ( p ) )-( ( a ) ) ) ) , unknown , '../../TPTP-v7.3.0/Problems/SYN/SYN306-1.p').
checkTheorem(syn , 344 , [ type:set ,b : (( type )) , a : (( type )) , bigY : (( type )) , p : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , bigU : (( type )) , bigZ : (( type )) , bigX : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) ] , ( ~( ( ~( ( ~( ~( ( ( ( p ) )-( ( bigX ) )-( ( bigZ ) )-( ( bigU ) ) ) \/ ( ( ~( ( ( p ) )-( ( bigX ) )-( ( bigY ) )-( ( bigU ) ) ) ) \/ ( ~( ( ( p ) )-( ( bigY ) )-( ( bigZ ) )-( ( bigU ) ) ) ) ) ) ) ) & ( ( ( p ) )-( ( bigX ) )-( ( bigX ) )-( ( a ) ) ) ) ) & ( ( ~( ( ( p ) )-( ( bigX ) )-( ( bigZ ) )-( ( bigU ) ) ) ) \/ ( ( ( ( p ) )-( ( bigX ) )-( ( bigY ) )-( ( bigU ) ) ) \/ ( ( ( p ) )-( ( bigY ) )-( ( bigZ ) )-( ( bigU ) ) ) ) ) ) ) & ( ~( ( ( p ) )-( ( bigX ) )-( ( bigX ) )-( ( b ) ) ) ) , unknown , '../../TPTP-v7.3.0/Problems/SYN/SYN307-1.p').
checkTheorem(syn , 345 , [ type:set ,r : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , e : (( ( type ) ) -> ( ( prop ) )) , bigV : (( type )) , g : (( ( type ) ) -> ( ( prop ) )) , b : (( type )) , h : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , bigZ : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , bigU : (( type )) , p : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , gf : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX : (( type )) , bigY : (( type )) , c : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , f : (( ( type ) ) -> ( ( prop ) )) , a : (( ( type ) ) -> ( ( prop ) )) ] , ( ~( ( ~( ( ~( ~( ( ( ( p ) )-( ( ( f ) )-( ( ( f ) )-( ( a ) ) ) )-( ( ( gf ) )-( ( c ) )-( ( ( gf ) )-( ( bigY ) )-( ( bigX ) ) ) ) ) \/ ( ( ( p ) )-( ( bigU ) )-( ( ( f ) )-( ( bigU ) ) ) ) ) ) ) & ( ( ~( ( ( p ) )-( ( ( f ) )-( ( ( h ) )-( ( bigX ) )-( ( bigY ) )-( ( bigZ ) ) ) )-( ( bigZ ) ) ) ) \/ ( ( ( ( g ) )-( ( b ) ) ) \/ ( ( ( ( g ) )-( ( bigV ) ) ) \/ ( ~( ( ( g ) )-( ( bigU ) ) ) ) ) ) ) ) ) & ( ( ( e ) )-( ( ( gf ) )-( ( ( h ) )-( ( bigZ ) )-( ( bigX ) )-( ( bigY ) ) )-( ( ( f ) )-( ( b ) ) ) ) ) ) ) & ( ( ( ( r ) )-( ( ( gf ) )-( ( bigV ) )-( ( ( gf ) )-( ( bigV ) )-( ( bigU ) ) ) )-( ( ( gf ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ( r ) )-( ( ( f ) )-( ( b ) ) )-( ( ( gf ) )-( ( ( f ) )-( ( ( f ) )-( ( a ) ) ) )-( ( ( gf ) )-( ( bigU ) )-( ( bigV ) ) ) ) ) ) , unknown , '../../TPTP-v7.3.0/Problems/SYN/SYN308-1.p').
checkTheorem(syn , 346 , [ type:set ,c : (( type )) , k : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , r : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , b : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , s : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , h : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , bigZ : (( type )) , bigY : (( type )) , p : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , g : (( ( type ) ) -> ( ( prop ) )) , a : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , f : (( ( type ) ) -> ( ( prop ) )) , bigX : (( ( type ) ) -> ( ( prop ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( ( p ) )-( ( ( f ) )-( ( bigX ) ) )-( ( a ) )-( ( ( g ) )-( ( bigX ) ) ) ) \/ ( ~( ( ( s ) )-( ( ( h ) )-( ( bigX ) )-( ( bigY ) )-( ( bigZ ) ) )-( ( a ) )-( ( ( h ) )-( ( bigX ) )-( ( bigY ) )-( ( bigZ ) ) ) ) ) ) ) ) & ( ( ( ( r ) )-( ( ( f ) )-( ( a ) ) )-( ( a ) )-( ( b ) ) ) \/ ( ( ( s ) )-( ( ( h ) )-( ( a ) )-( ( bigY ) )-( ( bigZ ) ) )-( ( b ) )-( ( ( h ) )-( ( a ) )-( ( bigY ) )-( ( bigZ ) ) ) ) ) ) ) & ( ( ( ( s ) )-( ( ( f ) )-( ( bigX ) ) )-( ( bigX ) )-( ( bigX ) ) ) \/ ( ( ~( ( ( p ) )-( ( ( k ) )-( ( bigX ) )-( ( bigY ) ) )-( ( ( k ) )-( ( bigX ) )-( ( bigY ) ) )-( ( b ) ) ) ) \/ ( ( ( p ) )-( ( ( f ) )-( ( bigX ) ) )-( ( ( g ) )-( ( bigX ) ) )-( ( ( g ) )-( ( bigX ) ) ) ) ) ) ) ) & ( ( ( ( s ) )-( ( ( f ) )-( ( bigX ) ) )-( ( bigX ) )-( ( bigX ) ) ) \/ ( ( ( p ) )-( ( ( f ) )-( ( bigY ) ) )-( ( bigY ) )-( ( ( f ) )-( ( bigY ) ) ) ) ) ) ) & ( ( ( p ) )-( ( bigX ) )-( ( bigX ) )-( ( bigX ) ) ) ) ) & ( ( ( p ) )-( ( a ) )-( ( b ) )-( ( c ) ) ) , unknown , '../../TPTP-v7.3.0/Problems/SYN/SYN309-1.p').
checkTheorem(syn , 347 , [ type:set ,c : (( type )) , b : (( type )) , a : (( type )) , f : (( ( type ) ) -> ( ( prop ) )) , g : (( ( type ) ) -> ( ( prop ) )) , p : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , bigX : (( type )) , bigX1 : (( type )) , bigX2 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ~( ( ( p ) )-( ( bigX2 ) )-( ( bigX1 ) )-( ( bigX ) ) ) ) \/ ( ( ( p ) )-( ( bigX ) )-( ( bigX1 ) )-( ( bigX2 ) ) ) ) ) ) & ( ( ~( ( ( p ) )-( ( bigX1 ) )-( ( bigX ) )-( ( bigX2 ) ) ) ) \/ ( ( ( p ) )-( ( bigX ) )-( ( bigX1 ) )-( ( bigX2 ) ) ) ) ) ) & ( ( ~( ( ( p ) )-( ( bigX ) )-( ( bigX1 ) )-( ( ( g ) )-( ( bigX2 ) ) ) ) ) \/ ( ( ( p ) )-( ( bigX ) )-( ( bigX1 ) )-( ( bigX2 ) ) ) ) ) ) & ( ( ~( ( ( p ) )-( ( ( f ) )-( ( bigX ) ) )-( ( bigX1 ) )-( ( bigX2 ) ) ) ) \/ ( ( ( p ) )-( ( bigX ) )-( ( bigX1 ) )-( ( bigX2 ) ) ) ) ) ) & ( ~( ( ( p ) )-( ( a ) )-( ( b ) )-( ( c ) ) ) ) ) ) & ( ( ( p ) )-( ( ( f ) )-( ( ( g ) )-( ( a ) ) ) )-( ( ( f ) )-( ( ( g ) )-( ( b ) ) ) )-( ( ( f ) )-( ( ( g ) )-( ( c ) ) ) ) ) , no , '../../TPTP-v7.3.0/Problems/SYN/SYN310-1.p').
