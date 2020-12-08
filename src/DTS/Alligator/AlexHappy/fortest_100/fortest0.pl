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












checkTheorem(syn , 1 , [ type:set ,p_5 : (( type )) , p_4 : (( type )) , p_3 : (( type )) , p_2 : (( type )) , p_1 : (( type )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( p_1 ) ) \/ ( ( ( p_2 ) ) \/ ( ( ( p_3 ) ) \/ ( ( ( p_4 ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ( p_1 ) ) \/ ( ( ( p_2 ) ) \/ ( ( ( p_3 ) ) \/ ( ( ( p_4 ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) ) ) & ( ( ( p_1 ) ) \/ ( ( ( p_2 ) ) \/ ( ( ( p_3 ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ( p_1 ) ) \/ ( ( ( p_2 ) ) \/ ( ( ( p_3 ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) ) ) & ( ( ( p_1 ) ) \/ ( ( ( p_2 ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ( p_4 ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ( p_1 ) ) \/ ( ( ( p_2 ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ( p_4 ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) ) ) & ( ( ( p_1 ) ) \/ ( ( ( p_2 ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ( p_1 ) ) \/ ( ( ( p_2 ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) ) ) & ( ( ( p_1 ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ( p_3 ) ) \/ ( ( ( p_4 ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ( p_1 ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ( p_3 ) ) \/ ( ( ( p_4 ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) ) ) & ( ( ( p_1 ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ( p_3 ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ( p_1 ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ( p_3 ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) ) ) & ( ( ( p_1 ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ( p_4 ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ( p_1 ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ( p_4 ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) ) ) & ( ( ( p_1 ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ( p_1 ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ( p_2 ) ) \/ ( ( ( p_3 ) ) \/ ( ( ( p_4 ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ( p_2 ) ) \/ ( ( ( p_3 ) ) \/ ( ( ( p_4 ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ( p_2 ) ) \/ ( ( ( p_3 ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ( p_2 ) ) \/ ( ( ( p_3 ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ( p_2 ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ( p_4 ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ( p_2 ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ( p_4 ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ( p_2 ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ( p_2 ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ( p_3 ) ) \/ ( ( ( p_4 ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ( p_3 ) ) \/ ( ( ( p_4 ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ( p_3 ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ( p_3 ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ( p_4 ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ( p_4 ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ( p_5 ) ) ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ~( ( p_2 ) ) ) \/ ( ( ~( ( p_3 ) ) ) \/ ( ( ~( ( p_4 ) ) ) \/ ( ~( ( p_5 ) ) ) ) ) ) ) , no , '../../TPTP-v7.3.0/Problems/SYN/SYN001-1.005.p').
checkTheorem(syn , 2 , [ type:set ,p : (( type )) ] , ( ( ~( ~( ( p ) ) ) ) -> ( ( p ) ) ) & ( ( ( p ) ) -> ( ~( ~( ( p ) ) ) ) ) , yes , '../../TPTP-v7.3.0/Problems/SYN/SYN001+1.p').




checkTheorem(syn , 3 , [ type:set ,f : (( ( type ) ) -> ( ( prop ) )) , p : (( ( type ) ) -> ( ( prop ) )) , bigX : (( ( type ) ) -> ( ( prop ) )) ] , ( ~( ~( ( ( ( p ) )-( ( bigX ) ) ) \/ ( ( ( p ) )-( ( ( f ) )-( ( ( f ) )-( ( ( f ) )-( ( ( f ) )-( ( ( f ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( p ) )-( ( bigX ) ) ) ) \/ ( ~( ( ( p ) )-( ( ( f ) )-( ( ( f ) )-( ( ( f ) )-( ( ( f ) )-( ( ( f ) )-( ( ( f ) )-( ( ( f ) )-( ( ( f ) )-( ( bigX ) ) ) ) ) ) ) ) ) ) ) ) ) , no , '../../TPTP-v7.3.0/Problems/SYN/SYN002-1.007.008.p').
checkTheorem(syn , 4 , [ type:set ,r : (( type )) , q : (( type )) , r_5 : (( type )) , r_4 : (( type )) , r_3 : (( type )) , r_2 : (( type )) , r_1 : (( type )) , p_6 : (( type )) , q_5 : (( type )) , p_5 : (( type )) , q_4 : (( type )) , p_4 : (( type )) , q_3 : (( type )) , p_3 : (( type )) , q_2 : (( type )) , p_2 : (( type )) , q_1 : (( type )) , p_1 : (( type )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ~( ( p_1 ) ) ) \/ ( ( ~( ( q_1 ) ) ) \/ ( ( p_2 ) ) ) ) ) ) & ( ( ~( ( p_2 ) ) ) \/ ( ( ~( ( q_2 ) ) ) \/ ( ( p_3 ) ) ) ) ) ) & ( ( ~( ( p_3 ) ) ) \/ ( ( ~( ( q_3 ) ) ) \/ ( ( p_4 ) ) ) ) ) ) & ( ( ~( ( p_4 ) ) ) \/ ( ( ~( ( q_4 ) ) ) \/ ( ( p_5 ) ) ) ) ) ) & ( ( ~( ( p_5 ) ) ) \/ ( ( ~( ( q_5 ) ) ) \/ ( ( p_6 ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ~( ( r_1 ) ) ) \/ ( ( p_2 ) ) ) ) ) ) & ( ( ~( ( p_2 ) ) ) \/ ( ( ~( ( r_2 ) ) ) \/ ( ( p_3 ) ) ) ) ) ) & ( ( ~( ( p_3 ) ) ) \/ ( ( ~( ( r_3 ) ) ) \/ ( ( p_4 ) ) ) ) ) ) & ( ( ~( ( p_4 ) ) ) \/ ( ( ~( ( r_4 ) ) ) \/ ( ( p_5 ) ) ) ) ) ) & ( ( ~( ( p_5 ) ) ) \/ ( ( ~( ( r_5 ) ) ) \/ ( ( p_6 ) ) ) ) ) ) & ( ( ~( ( q ) ) ) \/ ( ( q_1 ) ) ) ) ) & ( ( ~( ( q ) ) ) \/ ( ( q_2 ) ) ) ) ) & ( ( ~( ( q ) ) ) \/ ( ( q_3 ) ) ) ) ) & ( ( ~( ( q ) ) ) \/ ( ( q_4 ) ) ) ) ) & ( ( ~( ( q ) ) ) \/ ( ( q_5 ) ) ) ) ) & ( ( ~( ( r ) ) ) \/ ( ( r_1 ) ) ) ) ) & ( ( ~( ( r ) ) ) \/ ( ( r_2 ) ) ) ) ) & ( ( ~( ( r ) ) ) \/ ( ( r_3 ) ) ) ) ) & ( ( ~( ( r ) ) ) \/ ( ( r_4 ) ) ) ) ) & ( ( ~( ( r ) ) ) \/ ( ( r_5 ) ) ) ) ) & ( ( p_1 ) ) ) ) & ( ~( ( p_6 ) ) ) ) ) & ( ( q ) ) ) ) & ( ( r ) ) , no , '../../TPTP-v7.3.0/Problems/SYN/SYN003-1.006.p').
checkTheorem(syn , 5 , [ type:set ,q_7 : (( type )) , p_7 : (( type )) , q_6 : (( type )) , p_6 : (( type )) , q_5 : (( type )) , p_5 : (( type )) , q_4 : (( type )) , p_4 : (( type )) , q_3 : (( type )) , p_3 : (( type )) , q_2 : (( type )) , p_2 : (( type )) , q_1 : (( type )) , p_1 : (( type )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ~( ( p_1 ) ) ) \/ ( ( ~( ( q_1 ) ) ) \/ ( ( p_2 ) ) ) ) ) ) & ( ( ~( ( p_2 ) ) ) \/ ( ( ~( ( q_2 ) ) ) \/ ( ( p_3 ) ) ) ) ) ) & ( ( ~( ( p_3 ) ) ) \/ ( ( ~( ( q_3 ) ) ) \/ ( ( p_4 ) ) ) ) ) ) & ( ( ~( ( p_4 ) ) ) \/ ( ( ~( ( q_4 ) ) ) \/ ( ( p_5 ) ) ) ) ) ) & ( ( ~( ( p_5 ) ) ) \/ ( ( ~( ( q_5 ) ) ) \/ ( ( p_6 ) ) ) ) ) ) & ( ( ~( ( p_6 ) ) ) \/ ( ( ~( ( q_6 ) ) ) \/ ( ( p_7 ) ) ) ) ) ) & ( ( ~( ( p_1 ) ) ) \/ ( ( ~( ( q_1 ) ) ) \/ ( ( q_2 ) ) ) ) ) ) & ( ( ~( ( p_2 ) ) ) \/ ( ( ~( ( q_2 ) ) ) \/ ( ( q_3 ) ) ) ) ) ) & ( ( ~( ( p_3 ) ) ) \/ ( ( ~( ( q_3 ) ) ) \/ ( ( q_4 ) ) ) ) ) ) & ( ( ~( ( p_4 ) ) ) \/ ( ( ~( ( q_4 ) ) ) \/ ( ( q_5 ) ) ) ) ) ) & ( ( ~( ( p_5 ) ) ) \/ ( ( ~( ( q_5 ) ) ) \/ ( ( q_6 ) ) ) ) ) ) & ( ( ~( ( p_6 ) ) ) \/ ( ( ~( ( q_6 ) ) ) \/ ( ( q_7 ) ) ) ) ) ) & ( ( p_1 ) ) ) ) & ( ( q_1 ) ) ) ) & ( ( ~( ( p_7 ) ) ) \/ ( ~( ( q_7 ) ) ) ) , no , '../../TPTP-v7.3.0/Problems/SYN/SYN004-1.007.p').
