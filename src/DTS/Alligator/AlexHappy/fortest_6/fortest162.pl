
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
checkTheorem(syn , 942 , [ type:set ,q : (( ( type ) ) -> ( ( prop ) )) , p : (( ( type ) ) -> ( ( prop ) )) ] , ( pi(X:type,sigma(Y:type,( ( ( p ) )-( ( bigX ) ) ) & ( ( ( ( q ) )-( ( bigY ) ) ) \/ ( ( ( q ) )-( ( bigX ) ) ) ))) ) -> ( sigma(Z:type,( ( ( p ) )-( ( bigZ ) ) ) & ( ( ( q ) )-( ( bigZ ) ) )) ) , yes , '../../TPTP-v7.3.0/Problems/SYN/SYN733+1.p').
checkTheorem(syn , 943 , [ type:set ,bigX : (( type )) , bigW : (( type )) , ssPv3 : (( ( type ) ) -> ( ( prop ) )) , ssPv2 : (( ( type ) ) -> ( ( prop ) )) , ssPv1 : (( ( type ) ) -> ( ( prop ) )) , ssPv4 : (( ( type ) ) -> ( ( prop ) )) , bigV : (( ( type ) ) -> ( ( prop ) )) , ssRr : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , skf1 : (( ( type ) ) -> ( ( prop ) )) , bigU : (( ( type ) ) -> ( ( prop ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( ssRr ) )-( ( bigU ) )-( ( ( skf1 ) )-( ( bigU ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigU ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigU ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigU ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigU ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigU ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigU ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigV ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigU ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigV ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigU ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigU ) ) ) ) \/ ( ~( ( ( ssPv3 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigW ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigU ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigW ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigU ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv1 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv1 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigU ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigU ) ) ) ) \/ ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv1 ) )-( ( bigX ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigX ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigX ) ) ) \/ ( ( ( ssPv1 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigX ) ) ) ) ) ) ) ) ) , unknown , '../../TPTP-v7.3.0/Problems/SYN/SYN734-1.p').
checkTheorem(syn , 944 , [ type:set ,bigX : (( type )) , bigW : (( type )) , ssPv2 : (( ( type ) ) -> ( ( prop ) )) , ssPv4 : (( ( type ) ) -> ( ( prop ) )) , ssPv3 : (( ( type ) ) -> ( ( prop ) )) , ssPv1 : (( ( type ) ) -> ( ( prop ) )) , bigV : (( ( type ) ) -> ( ( prop ) )) , ssRr : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , skf1 : (( ( type ) ) -> ( ( prop ) )) , bigU : (( ( type ) ) -> ( ( prop ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( ssRr ) )-( ( bigU ) )-( ( ( skf1 ) )-( ( bigU ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigU ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigU ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigU ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigU ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigW ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigU ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigW ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigU ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv1 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigV ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigV ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigU ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigW ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigX ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv1 ) )-( ( bigX ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigX ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigX ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigX ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigX ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigX ) ) ) \/ ( ( ( ssPv1 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigX ) ) ) ) ) ) ) ) ) , unknown , '../../TPTP-v7.3.0/Problems/SYN/SYN735-1.p').
checkTheorem(syn , 945 , [ type:set ,bigX1 : (( type )) , bigZ : (( type )) , bigY : (( type )) , bigX : (( type )) , ssPv3 : (( ( type ) ) -> ( ( prop ) )) , ssPv2 : (( ( type ) ) -> ( ( prop ) )) , ssPv1 : (( ( type ) ) -> ( ( prop ) )) , bigW : (( ( type ) ) -> ( ( prop ) )) , ssPv4 : (( ( type ) ) -> ( ( prop ) )) , bigV : (( ( type ) ) -> ( ( prop ) )) , ssRr : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , skf1 : (( ( type ) ) -> ( ( prop ) )) , bigU : (( ( type ) ) -> ( ( prop ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( ssRr ) )-( ( bigU ) )-( ( ( skf1 ) )-( ( bigU ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigW ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigX ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigX ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigX ) ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigY ) ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigY ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigX ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigY ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigV ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigY ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigY ) ) ) \/ ( ( ( ssPv1 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigY ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigW ) ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigY ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigZ ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigY ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigZ ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigZ ) ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigW ) ) ) ) \/ ( ( ( ssPv1 ) )-( ( bigZ ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigW ) ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigZ ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigZ ) ) ) ) \/ ( ~( ( ( ssPv4 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigZ ) )-( ( bigX1 ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigY ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigX1 ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigZ ) )-( ( bigX1 ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigX1 ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigZ ) )-( ( bigX1 ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigX1 ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigZ ) )-( ( bigX1 ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigW ) ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigX1 ) ) ) ) ) ) ) ) ) ) ) ) , unknown , '../../TPTP-v7.3.0/Problems/SYN/SYN736-1.p').
checkTheorem(syn , 946 , [ type:set ,bigX1 : (( type )) , bigZ : (( type )) , bigY : (( type )) , ssPv2 : (( ( type ) ) -> ( ( prop ) )) , bigX : (( ( type ) ) -> ( ( prop ) )) , ssPv1 : (( ( type ) ) -> ( ( prop ) )) , ssPv4 : (( ( type ) ) -> ( ( prop ) )) , ssPv3 : (( ( type ) ) -> ( ( prop ) )) , bigW : (( ( type ) ) -> ( ( prop ) )) , bigV : (( ( type ) ) -> ( ( prop ) )) , ssRr : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , skf1 : (( ( type ) ) -> ( ( prop ) )) , bigU : (( ( type ) ) -> ( ( prop ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( ssRr ) )-( ( bigU ) )-( ( ( skf1 ) )-( ( bigU ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigX ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigX ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigV ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigV ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigY ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigY ) ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigY ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigY ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigY ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigY ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigY ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigY ) ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigV ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigY ) ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigW ) ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigY ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigZ ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigW ) ) ) ) \/ ( ~( ( ( ssPv2 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigY ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigZ ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigZ ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigV ) ) ) \/ ( ( ( ssPv1 ) )-( ( bigY ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigY ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigZ ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigZ ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigY ) ) ) \/ ( ( ( ssPv1 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigX ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigY ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigZ ) )-( ( bigX1 ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigW ) )-( ( bigZ ) ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigX1 ) ) ) \/ ( ( ( ssPv3 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) ) ) , unknown , '../../TPTP-v7.3.0/Problems/SYN/SYN737-1.p').
checkTheorem(syn , 947 , [ type:set ,bigX : (( type )) , bigW : (( type )) , ssPv6 : (( ( type ) ) -> ( ( prop ) )) , ssPv7 : (( ( type ) ) -> ( ( prop ) )) , ssPv1 : (( ( type ) ) -> ( ( prop ) )) , ssPv2 : (( ( type ) ) -> ( ( prop ) )) , ssPv5 : (( ( type ) ) -> ( ( prop ) )) , ssPv4 : (( ( type ) ) -> ( ( prop ) )) , ssPv8 : (( ( type ) ) -> ( ( prop ) )) , ssPv3 : (( ( type ) ) -> ( ( prop ) )) , bigV : (( ( type ) ) -> ( ( prop ) )) , ssRr : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , skf1 : (( ( type ) ) -> ( ( prop ) )) , bigU : (( ( type ) ) -> ( ( prop ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( ssRr ) )-( ( bigU ) )-( ( ( skf1 ) )-( ( bigU ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv8 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv4 ) )-( ( bigU ) ) ) \/ ( ( ( ssPv5 ) )-( ( bigU ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv5 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv8 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv2 ) )-( ( bigV ) ) ) \/ ( ( ( ssPv2 ) )-( ( bigU ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv1 ) )-( ( bigU ) ) ) \/ ( ( ( ssPv7 ) )-( ( bigU ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv7 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv6 ) )-( ( bigU ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv5 ) )-( ( bigW ) ) ) \/ ( ( ( ( ssPv6 ) )-( ( bigU ) ) ) \/ ( ( ( ssPv8 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv5 ) )-( ( bigV ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv5 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv8 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigW ) ) ) ) \/ ( ( ( ( ssPv5 ) )-( ( bigU ) ) ) \/ ( ( ( ssPv7 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv6 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv5 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv6 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv7 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv8 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv8 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv7 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv8 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv7 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv6 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv7 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv7 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigU ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigW ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv8 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv5 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv2 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv7 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ~( ( ( ssPv8 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv7 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ( ( ssPv8 ) )-( ( bigW ) ) ) \/ ( ( ( ( ssPv3 ) )-( ( bigX ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigU ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv6 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv8 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv6 ) )-( ( bigX ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv5 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv5 ) )-( ( bigU ) ) ) ) \/ ( ( ( ( ssPv6 ) )-( ( bigW ) ) ) \/ ( ( ( ssPv4 ) )-( ( bigX ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv6 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv4 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv5 ) )-( ( bigX ) ) ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssPv3 ) )-( ( bigV ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssPv1 ) )-( ( bigW ) ) ) ) \/ ( ( ~( ( ( ssRr ) )-( ( bigU ) )-( ( bigX ) ) ) ) \/ ( ( ~( ( ( ssPv7 ) )-( ( bigU ) ) ) ) \/ ( ( ( ssPv7 ) )-( ( bigX ) ) ) ) ) ) ) ) ) , unknown , '../../TPTP-v7.3.0/Problems/SYN/SYN738-1.p').
