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
checkTheorem(syn , 114 , [ type:set ,big_f : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , f : (( ( type ) ) -> ( ( prop ) )) ] , pi(X:type,( ( ( ( big_f ) )-( ( bigX ) )-( ( ( f ) )-( ( bigX ) ) ) ) -> ( sigma(Y:type,( pi(Z:type,( ( ( big_f ) )-( ( bigZ ) )-( ( bigY ) ) ) -> ( ( ( big_f ) )-( ( bigZ ) )-( ( ( f ) )-( ( bigX ) ) ) )) ) & ( ( ( big_f ) )-( ( bigX ) )-( ( bigY ) ) )) ) ) & ( ( sigma(Y:type,( pi(Z:type,( ( ( big_f ) )-( ( bigZ ) )-( ( bigY ) ) ) -> ( ( ( big_f ) )-( ( bigZ ) )-( ( ( f ) )-( ( bigX ) ) ) )) ) & ( ( ( big_f ) )-( ( bigX ) )-( ( bigY ) ) )) ) -> ( ( ( big_f ) )-( ( bigX ) )-( ( ( f ) )-( ( bigX ) ) ) ) )) , yes , '../../TPTP-v7.3.0/Problems/SYN/SYN082+1.p').
checkTheorem(syn , 115 , [ type:set ,d : (( type )) , c : (( type )) , b : (( type )) , a : (( type )) , f : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigZ : (( type )) , bigY : (( type )) , bigX : (( type )) , axiom0 : (( ( ( ( f ) )-( ( bigX ) )-( ( ( f ) )-( ( bigY ) )-( ( bigZ ) ) ) ) -> ( ( ( f ) )-( ( ( f ) )-( ( bigX ) )-( ( bigY ) ) )-( ( bigZ ) ) ) ) & ( ( ( ( f ) )-( ( ( f ) )-( ( bigX ) )-( ( bigY ) ) )-( ( bigZ ) ) ) -> ( ( ( f ) )-( ( bigX ) )-( ( ( f ) )-( ( bigY ) )-( ( bigZ ) ) ) ) )) ] , ~( ~( ( ( ( ( f ) )-( ( a ) )-( ( ( f ) )-( ( b ) )-( ( ( f ) )-( ( c ) )-( ( d ) ) ) ) ) -> ( ( ( f ) )-( ( ( f ) )-( ( ( f ) )-( ( a ) )-( ( b ) ) )-( ( c ) ) )-( ( d ) ) ) ) & ( ( ( ( f ) )-( ( ( f ) )-( ( ( f ) )-( ( a ) )-( ( b ) ) )-( ( c ) ) )-( ( d ) ) ) -> ( ( ( f ) )-( ( a ) )-( ( ( f ) )-( ( b ) )-( ( ( f ) )-( ( c ) )-( ( d ) ) ) ) ) ) ) ) , no , '../../TPTP-v7.3.0/Problems/SYN/SYN083-1.p').
checkTheorem(syn , 116 , [ type:set ,bigW : (( type )) , bigZ : (( type )) , bigY : (( type )) , bigX : (( type )) , f : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , axiom0 : (( ( pi(X:type,pi(Y:type,pi(Z:type,( ( f ) )-( ( bigX ) )-( ( ( f ) )-( ( bigY ) )-( ( bigZ ) ) )))) ) -> ( ( ( f ) )-( ( ( f ) )-( ( bigX ) )-( ( bigY ) ) )-( ( bigZ ) ) ) ) & ( ( ( ( f ) )-( ( ( f ) )-( ( bigX ) )-( ( bigY ) ) )-( ( bigZ ) ) ) -> ( pi(X:type,pi(Y:type,pi(Z:type,( ( f ) )-( ( bigX ) )-( ( ( f ) )-( ( bigY ) )-( ( bigZ ) ) )))) ) )) ] , ( ( pi(X:type,pi(Y:type,pi(Z:type,pi(W:type,( ( f ) )-( ( bigX ) )-( ( ( f ) )-( ( bigY ) )-( ( ( f ) )-( ( bigZ ) )-( ( bigW ) ) ) ))))) ) -> ( ( ( f ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX ) )-( ( bigY ) ) )-( ( bigZ ) ) )-( ( bigW ) ) ) ) & ( ( ( ( f ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX ) )-( ( bigY ) ) )-( ( bigZ ) ) )-( ( bigW ) ) ) -> ( pi(X:type,pi(Y:type,pi(Z:type,pi(W:type,( ( f ) )-( ( bigX ) )-( ( ( f ) )-( ( bigY ) )-( ( ( f ) )-( ( bigZ ) )-( ( bigW ) ) ) ))))) ) ) , yes , '../../TPTP-v7.3.0/Problems/SYN/SYN083+1.p').
checkTheorem(syn , 117 , [ type:set ,b : (( type )) , c : (( type )) , bigX : (( type )) , f : (( ( type ) ) -> ( ( prop ) )) , bigY : (( ( type ) ) -> ( ( prop ) )) , big_p : (( ( type ) ) -> ( ( prop ) )) , a : (( ( type ) ) -> ( ( prop ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( big_p ) )-( ( a ) ) ) ) ) & ( ( ~( ( ( big_p ) )-( ( ( f ) )-( ( bigY ) ) ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) \/ ( ( ( ( big_p ) )-( ( bigX ) ) ) \/ ( ~( ( ( big_p ) )-( ( a ) ) ) ) ) ) ) ) ) & ( ( ( ( big_p ) )-( ( bigY ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) \/ ( ( ( ( big_p ) )-( ( bigX ) ) ) \/ ( ~( ( ( big_p ) )-( ( a ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( big_p ) )-( ( c ) ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( c ) ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) \/ ( ( ( ( big_p ) )-( ( bigX ) ) ) \/ ( ~( ( ( big_p ) )-( ( a ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( c ) ) ) ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) \/ ( ( ( ( big_p ) )-( ( bigX ) ) ) \/ ( ~( ( ( big_p ) )-( ( a ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( c ) ) ) ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) \/ ( ( ~( ( ( big_p ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) \/ ( ~( ( ( big_p ) )-( ( a ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( big_p ) )-( ( c ) ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( c ) ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) \/ ( ( ~( ( ( big_p ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) \/ ( ~( ( ( big_p ) )-( ( a ) ) ) ) ) ) ) ) ) ) & ( ( ( ( big_p ) )-( ( bigY ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) \/ ( ( ~( ( ( big_p ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) \/ ( ~( ( ( big_p ) )-( ( a ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( big_p ) )-( ( ( f ) )-( ( bigY ) ) ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) \/ ( ( ~( ( ( big_p ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) \/ ( ~( ( ( big_p ) )-( ( a ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( c ) ) ) ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( b ) ) ) ) \/ ( ~( ( ( big_p ) )-( ( b ) ) ) ) ) ) ) ) & ( ( ~( ( ( big_p ) )-( ( c ) ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( c ) ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( b ) ) ) ) \/ ( ~( ( ( big_p ) )-( ( b ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( big_p ) )-( ( a ) ) ) ) \/ ( ( ( ( big_p ) )-( ( bigY ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( b ) ) ) ) \/ ( ~( ( ( big_p ) )-( ( b ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( big_p ) )-( ( a ) ) ) ) \/ ( ( ~( ( ( big_p ) )-( ( ( f ) )-( ( bigY ) ) ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( b ) ) ) ) \/ ( ~( ( ( big_p ) )-( ( b ) ) ) ) ) ) ) , unknown , '../../TPTP-v7.3.0/Problems/SYN/SYN084-1.p').
checkTheorem(syn , 118 , [ type:set ,f : (( ( type ) ) -> ( ( prop ) )) , big_p : (( ( type ) ) -> ( ( prop ) )) , a : (( ( type ) ) -> ( ( prop ) )) ] , ( ( pi(X:type,( ( ( ( big_p ) )-( ( a ) ) ) & ( ( ( ( big_p ) )-( ( bigX ) ) ) -> ( ( ( big_p ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) ) -> ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX ) ) ) ) )) ) -> ( pi(X1:type,( ( ~( ( ( big_p ) )-( ( a ) ) ) ) \/ ( ( ( ( big_p ) )-( ( bigX1 ) ) ) \/ ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX1 ) ) ) ) ) ) ) & ( ( ~( ( ( big_p ) )-( ( a ) ) ) ) \/ ( ( ~( ( ( big_p ) )-( ( ( f ) )-( ( bigX1 ) ) ) ) ) \/ ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX1 ) ) ) ) ) ) )) ) ) & ( ( pi(X1:type,( ( ~( ( ( big_p ) )-( ( a ) ) ) ) \/ ( ( ( ( big_p ) )-( ( bigX1 ) ) ) \/ ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX1 ) ) ) ) ) ) ) & ( ( ~( ( ( big_p ) )-( ( a ) ) ) ) \/ ( ( ~( ( ( big_p ) )-( ( ( f ) )-( ( bigX1 ) ) ) ) ) \/ ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX1 ) ) ) ) ) ) )) ) -> ( pi(X:type,( ( ( ( big_p ) )-( ( a ) ) ) & ( ( ( ( big_p ) )-( ( bigX ) ) ) -> ( ( ( big_p ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) ) -> ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX ) ) ) ) )) ) ) , yes , '../../TPTP-v7.3.0/Problems/SYN/SYN084+1.p').
checkTheorem(syn , 119 , [ type:set ,sk2 : (( type )) , sk1 : (( type )) , f : (( ( type ) ) -> ( ( prop ) )) , bigA : (( ( type ) ) -> ( ( prop ) )) , big_p : (( ( type ) ) -> ( ( prop ) )) , a : (( ( type ) ) -> ( ( prop ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( big_p ) )-( ( a ) ) ) ) ) & ( ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigA ) ) ) ) ) \/ ( ( ( ( big_p ) )-( ( bigA ) ) ) \/ ( ~( ( ( big_p ) )-( ( a ) ) ) ) ) ) ) ) & ( ( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( bigA ) ) ) ) ) \/ ( ( ~( ( ( big_p ) )-( ( ( f ) )-( ( bigA ) ) ) ) ) \/ ( ~( ( ( big_p ) )-( ( a ) ) ) ) ) ) ) ) & ( ( ~( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( sk1 ) ) ) ) ) ) \/ ( ~( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( sk2 ) ) ) ) ) ) ) ) ) & ( ( ( ( big_p ) )-( ( ( f ) )-( ( sk1 ) ) ) ) \/ ( ( ( ( big_p ) )-( ( ( f ) )-( ( sk2 ) ) ) ) \/ ( ( ~( ( ( big_p ) )-( ( sk1 ) ) ) ) \/ ( ~( ( ( big_p ) )-( ( sk2 ) ) ) ) ) ) ) ) ) & ( ( ( ( big_p ) )-( ( ( f ) )-( ( sk1 ) ) ) ) \/ ( ( ~( ( ( big_p ) )-( ( sk1 ) ) ) ) \/ ( ~( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( sk2 ) ) ) ) ) ) ) ) ) ) & ( ( ( ( big_p ) )-( ( ( f ) )-( ( sk2 ) ) ) ) \/ ( ( ~( ( ( big_p ) )-( ( sk2 ) ) ) ) \/ ( ~( ( ( big_p ) )-( ( ( f ) )-( ( ( f ) )-( ( sk1 ) ) ) ) ) ) ) ) , no , '../../TPTP-v7.3.0/Problems/SYN/SYN084-2.p').
