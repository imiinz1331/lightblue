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
checkTheorem(syn , 780 , [ type:set ,bigX34 : (( type )) , bigX33 : (( type )) , bigX27 : (( type )) , bigX26 : (( type )) , f8 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX25 : (( type )) , bigX24 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX30 : (( type )) , bigX31 : (( type )) , bigX29 : (( type )) , bigX28 : (( type )) , bigX5 : (( type )) , bigX6 : (( type )) , bigX4 : (( type )) , bigX3 : (( type )) , bigX9 : (( type )) , bigX8 : (( type )) , bigX14 : (( type )) , bigX13 : (( type )) , bigX23 : (( type )) , bigX22 : (( type )) , bigX2 : (( type )) , bigX1 : (( type )) , bigX16 : (( type )) , bigX15 : (( type )) , bigX19 : (( type )) , bigX18 : (( type )) , bigX11 : (( type )) , bigX10 : (( type )) , bigX32 : (( type )) , f6 : (( ( type ) ) -> ( ( prop ) )) , bigX20 : (( ( type ) ) -> ( ( prop ) )) , f4 : (( ( type ) ) -> ( ( prop ) )) , bigX17 : (( ( type ) ) -> ( ( prop ) )) , f5 : (( ( type ) ) -> ( ( prop ) )) , c16 : (( ( type ) ) -> ( ( prop ) )) , p11 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , c15 : (( type )) , c14 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p9 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , c13 : (( type )) , c12 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p2 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX7 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p3 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX12 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p7 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX21 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p10 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX0 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( p10 ) )-( ( bigX0 ) )-( ( bigX0 ) ) ) ) ) & ( ( ( p7 ) )-( ( bigX21 ) )-( ( bigX21 ) ) ) ) ) & ( ( ( p3 ) )-( ( bigX12 ) )-( ( bigX12 ) ) ) ) ) & ( ( ( p2 ) )-( ( bigX7 ) )-( ( bigX7 ) ) ) ) ) & ( ( ( p9 ) )-( ( c12 ) )-( ( c13 ) ) ) ) ) & ( ( ( p11 ) )-( ( c14 ) )-( ( c15 ) ) ) ) ) & ( ( ( p2 ) )-( ( c16 ) )-( ( ( f5 ) )-( ( c14 ) ) ) ) ) ) & ( ( ( p3 ) )-( ( ( f4 ) )-( ( ( f5 ) )-( ( bigX17 ) ) ) )-( ( ( f4 ) )-( ( bigX17 ) ) ) ) ) ) & ( ( ( p3 ) )-( ( ( f6 ) )-( ( ( f5 ) )-( ( bigX20 ) ) ) )-( ( ( f6 ) )-( ( bigX20 ) ) ) ) ) ) & ( ( ( ( p9 ) )-( ( bigX32 ) )-( ( c13 ) ) ) \/ ( ~( ( ( p9 ) )-( ( bigX32 ) )-( ( ( f4 ) )-( ( c16 ) ) ) ) ) ) ) ) & ( ( ( ( p9 ) )-( ( bigX32 ) )-( ( ( f4 ) )-( ( c16 ) ) ) ) \/ ( ~( ( ( p9 ) )-( ( bigX32 ) )-( ( c13 ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( ( f5 ) )-( ( bigX10 ) ) )-( ( ( f5 ) )-( ( bigX11 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX10 ) )-( ( bigX11 ) ) ) ) ) ) ) & ( ( ( ( p3 ) )-( ( ( f6 ) )-( ( bigX18 ) ) )-( ( ( f6 ) )-( ( bigX19 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX18 ) )-( ( bigX19 ) ) ) ) ) ) ) & ( ( ( ( p3 ) )-( ( ( f4 ) )-( ( bigX15 ) ) )-( ( ( f4 ) )-( ( bigX16 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX15 ) )-( ( bigX16 ) ) ) ) ) ) ) & ( ( ( ( p10 ) )-( ( bigX1 ) )-( ( bigX2 ) ) ) \/ ( ( ~( ( ( p10 ) )-( ( bigX0 ) )-( ( bigX1 ) ) ) ) \/ ( ~( ( ( p10 ) )-( ( bigX0 ) )-( ( bigX2 ) ) ) ) ) ) ) ) & ( ( ( ( p7 ) )-( ( bigX22 ) )-( ( bigX23 ) ) ) \/ ( ( ~( ( ( p7 ) )-( ( bigX21 ) )-( ( bigX22 ) ) ) ) \/ ( ~( ( ( p7 ) )-( ( bigX21 ) )-( ( bigX23 ) ) ) ) ) ) ) ) & ( ( ( ( p3 ) )-( ( bigX13 ) )-( ( bigX14 ) ) ) \/ ( ( ~( ( ( p3 ) )-( ( bigX12 ) )-( ( bigX13 ) ) ) ) \/ ( ~( ( ( p3 ) )-( ( bigX12 ) )-( ( bigX14 ) ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( bigX8 ) )-( ( bigX9 ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX7 ) )-( ( bigX8 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX7 ) )-( ( bigX9 ) ) ) ) ) ) ) ) & ( ( ( ( p11 ) )-( ( bigX3 ) )-( ( bigX4 ) ) ) \/ ( ( ~( ( ( p11 ) )-( ( bigX6 ) )-( ( bigX5 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX6 ) )-( ( bigX3 ) ) ) ) \/ ( ~( ( ( p10 ) )-( ( bigX5 ) )-( ( bigX4 ) ) ) ) ) ) ) ) ) & ( ( ( ( p9 ) )-( ( bigX28 ) )-( ( bigX29 ) ) ) \/ ( ( ~( ( ( p7 ) )-( ( bigX31 ) )-( ( bigX28 ) ) ) ) \/ ( ( ~( ( ( p9 ) )-( ( bigX31 ) )-( ( bigX30 ) ) ) ) \/ ( ~( ( ( p3 ) )-( ( bigX30 ) )-( ( bigX29 ) ) ) ) ) ) ) ) ) & ( ( ( ( p7 ) )-( ( ( f8 ) )-( ( bigX24 ) )-( ( bigX25 ) ) )-( ( ( f8 ) )-( ( bigX26 ) )-( ( bigX27 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX25 ) )-( ( bigX27 ) ) ) ) \/ ( ~( ( ( p3 ) )-( ( bigX24 ) )-( ( bigX26 ) ) ) ) ) ) ) ) & ( ( ( ( p9 ) )-( ( ( f8 ) )-( ( bigX33 ) )-( ( bigX34 ) ) )-( ( bigX33 ) ) ) \/ ( ( ( ( p9 ) )-( ( ( f8 ) )-( ( bigX33 ) )-( ( bigX34 ) ) )-( ( ( f4 ) )-( ( bigX34 ) ) ) ) \/ ( ( ~( ( ( p11 ) )-( ( bigX34 ) )-( ( c15 ) ) ) ) \/ ( ~( ( ( p9 ) )-( ( c12 ) )-( ( bigX33 ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( p11 ) )-( ( bigX34 ) )-( ( c15 ) ) ) ) \/ ( ( ~( ( ( p9 ) )-( ( c12 ) )-( ( bigX33 ) ) ) ) \/ ( ( ~( ( ( p9 ) )-( ( ( f8 ) )-( ( bigX33 ) )-( ( bigX34 ) ) )-( ( bigX33 ) ) ) ) \/ ( ~( ( ( p9 ) )-( ( ( f8 ) )-( ( bigX33 ) )-( ( bigX34 ) ) )-( ( ( f4 ) )-( ( bigX34 ) ) ) ) ) ) ) ) , no , '../../TPTP-v7.3.0/Problems/SYN/SYN578-1.p').
checkTheorem(syn , 781 , [ type:set ,bigX34 : (( type )) , bigX33 : (( type )) , bigX27 : (( type )) , bigX26 : (( type )) , f8 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX25 : (( type )) , bigX24 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX30 : (( type )) , bigX31 : (( type )) , bigX29 : (( type )) , bigX28 : (( type )) , bigX5 : (( type )) , bigX6 : (( type )) , bigX4 : (( type )) , bigX3 : (( type )) , bigX9 : (( type )) , bigX8 : (( type )) , bigX14 : (( type )) , bigX13 : (( type )) , bigX23 : (( type )) , bigX22 : (( type )) , bigX2 : (( type )) , bigX1 : (( type )) , bigX16 : (( type )) , bigX15 : (( type )) , bigX19 : (( type )) , bigX18 : (( type )) , bigX11 : (( type )) , bigX10 : (( type )) , bigX32 : (( type )) , f6 : (( ( type ) ) -> ( ( prop ) )) , bigX20 : (( ( type ) ) -> ( ( prop ) )) , f4 : (( ( type ) ) -> ( ( prop ) )) , bigX17 : (( ( type ) ) -> ( ( prop ) )) , f5 : (( ( type ) ) -> ( ( prop ) )) , c16 : (( ( type ) ) -> ( ( prop ) )) , p11 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , c15 : (( type )) , c14 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p9 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , c13 : (( type )) , c12 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p2 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX7 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p3 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX12 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p7 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX21 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p10 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX0 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( p10 ) )-( ( bigX0 ) )-( ( bigX0 ) ) ) ) ) & ( ( ( p7 ) )-( ( bigX21 ) )-( ( bigX21 ) ) ) ) ) & ( ( ( p3 ) )-( ( bigX12 ) )-( ( bigX12 ) ) ) ) ) & ( ( ( p2 ) )-( ( bigX7 ) )-( ( bigX7 ) ) ) ) ) & ( ( ( p9 ) )-( ( c12 ) )-( ( c13 ) ) ) ) ) & ( ( ( p11 ) )-( ( c14 ) )-( ( c15 ) ) ) ) ) & ( ( ( p2 ) )-( ( c16 ) )-( ( ( f5 ) )-( ( c14 ) ) ) ) ) ) & ( ( ( p3 ) )-( ( ( f4 ) )-( ( ( f5 ) )-( ( bigX17 ) ) ) )-( ( ( f4 ) )-( ( bigX17 ) ) ) ) ) ) & ( ( ( p3 ) )-( ( ( f6 ) )-( ( ( f5 ) )-( ( bigX20 ) ) ) )-( ( ( f6 ) )-( ( bigX20 ) ) ) ) ) ) & ( ( ( ( p9 ) )-( ( bigX32 ) )-( ( c13 ) ) ) \/ ( ~( ( ( p9 ) )-( ( bigX32 ) )-( ( ( f6 ) )-( ( c16 ) ) ) ) ) ) ) ) & ( ( ( ( p9 ) )-( ( bigX32 ) )-( ( ( f6 ) )-( ( c16 ) ) ) ) \/ ( ~( ( ( p9 ) )-( ( bigX32 ) )-( ( c13 ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( ( f5 ) )-( ( bigX10 ) ) )-( ( ( f5 ) )-( ( bigX11 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX10 ) )-( ( bigX11 ) ) ) ) ) ) ) & ( ( ( ( p3 ) )-( ( ( f6 ) )-( ( bigX18 ) ) )-( ( ( f6 ) )-( ( bigX19 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX18 ) )-( ( bigX19 ) ) ) ) ) ) ) & ( ( ( ( p3 ) )-( ( ( f4 ) )-( ( bigX15 ) ) )-( ( ( f4 ) )-( ( bigX16 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX15 ) )-( ( bigX16 ) ) ) ) ) ) ) & ( ( ( ( p10 ) )-( ( bigX1 ) )-( ( bigX2 ) ) ) \/ ( ( ~( ( ( p10 ) )-( ( bigX0 ) )-( ( bigX1 ) ) ) ) \/ ( ~( ( ( p10 ) )-( ( bigX0 ) )-( ( bigX2 ) ) ) ) ) ) ) ) & ( ( ( ( p7 ) )-( ( bigX22 ) )-( ( bigX23 ) ) ) \/ ( ( ~( ( ( p7 ) )-( ( bigX21 ) )-( ( bigX22 ) ) ) ) \/ ( ~( ( ( p7 ) )-( ( bigX21 ) )-( ( bigX23 ) ) ) ) ) ) ) ) & ( ( ( ( p3 ) )-( ( bigX13 ) )-( ( bigX14 ) ) ) \/ ( ( ~( ( ( p3 ) )-( ( bigX12 ) )-( ( bigX13 ) ) ) ) \/ ( ~( ( ( p3 ) )-( ( bigX12 ) )-( ( bigX14 ) ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( bigX8 ) )-( ( bigX9 ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX7 ) )-( ( bigX8 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX7 ) )-( ( bigX9 ) ) ) ) ) ) ) ) & ( ( ( ( p11 ) )-( ( bigX3 ) )-( ( bigX4 ) ) ) \/ ( ( ~( ( ( p11 ) )-( ( bigX6 ) )-( ( bigX5 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX6 ) )-( ( bigX3 ) ) ) ) \/ ( ~( ( ( p10 ) )-( ( bigX5 ) )-( ( bigX4 ) ) ) ) ) ) ) ) ) & ( ( ( ( p9 ) )-( ( bigX28 ) )-( ( bigX29 ) ) ) \/ ( ( ~( ( ( p7 ) )-( ( bigX31 ) )-( ( bigX28 ) ) ) ) \/ ( ( ~( ( ( p9 ) )-( ( bigX31 ) )-( ( bigX30 ) ) ) ) \/ ( ~( ( ( p3 ) )-( ( bigX30 ) )-( ( bigX29 ) ) ) ) ) ) ) ) ) & ( ( ( ( p7 ) )-( ( ( f8 ) )-( ( bigX24 ) )-( ( bigX25 ) ) )-( ( ( f8 ) )-( ( bigX26 ) )-( ( bigX27 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX25 ) )-( ( bigX27 ) ) ) ) \/ ( ~( ( ( p3 ) )-( ( bigX24 ) )-( ( bigX26 ) ) ) ) ) ) ) ) & ( ( ( ( p9 ) )-( ( ( f8 ) )-( ( bigX33 ) )-( ( bigX34 ) ) )-( ( bigX33 ) ) ) \/ ( ( ( ( p9 ) )-( ( ( f8 ) )-( ( bigX33 ) )-( ( bigX34 ) ) )-( ( ( f6 ) )-( ( bigX34 ) ) ) ) \/ ( ( ~( ( ( p11 ) )-( ( bigX34 ) )-( ( c15 ) ) ) ) \/ ( ~( ( ( p9 ) )-( ( c12 ) )-( ( bigX33 ) ) ) ) ) ) ) ) ) & ( ( ~( ( ( p11 ) )-( ( bigX34 ) )-( ( c15 ) ) ) ) \/ ( ( ~( ( ( p9 ) )-( ( c12 ) )-( ( bigX33 ) ) ) ) \/ ( ( ~( ( ( p9 ) )-( ( ( f8 ) )-( ( bigX33 ) )-( ( bigX34 ) ) )-( ( bigX33 ) ) ) ) \/ ( ~( ( ( p9 ) )-( ( ( f8 ) )-( ( bigX33 ) )-( ( bigX34 ) ) )-( ( ( f6 ) )-( ( bigX34 ) ) ) ) ) ) ) ) , no , '../../TPTP-v7.3.0/Problems/SYN/SYN579-1.p').
checkTheorem(syn , 782 , [ type:set ,bigX27 : (( type )) , bigX26 : (( type )) , bigX25 : (( type )) , bigX24 : (( type )) , bigX23 : (( type )) , bigX22 : (( type )) , bigX5 : (( type )) , bigX6 : (( type )) , bigX4 : (( type )) , bigX3 : (( type )) , bigX34 : (( type )) , bigX35 : (( type )) , bigX33 : (( type )) , bigX32 : (( type )) , bigX30 : (( type )) , bigX31 : (( type )) , bigX29 : (( type )) , bigX28 : (( type )) , bigX9 : (( type )) , bigX8 : (( type )) , bigX14 : (( type )) , bigX13 : (( type )) , bigX21 : (( type )) , bigX20 : (( type )) , bigX2 : (( type )) , bigX1 : (( type )) , bigX16 : (( type )) , f5 : (( ( type ) ) -> ( ( prop ) )) , bigX15 : (( ( type ) ) -> ( ( prop ) )) , bigX11 : (( type )) , bigX10 : (( type )) , bigX18 : (( type )) , f6 : (( ( type ) ) -> ( ( prop ) )) , bigX17 : (( ( type ) ) -> ( ( prop ) )) , p8 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , f3 : (( ( type ) ) -> ( ( prop ) )) , c14 : (( ( type ) ) -> ( ( prop ) )) , p9 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , c13 : (( type )) , c12 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p11 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , c16 : (( type )) , c15 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p2 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX7 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p4 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX12 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p7 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX19 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p10 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX0 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( p10 ) )-( ( bigX0 ) )-( ( bigX0 ) ) ) ) ) & ( ( ( p7 ) )-( ( bigX19 ) )-( ( bigX19 ) ) ) ) ) & ( ( ( p4 ) )-( ( bigX12 ) )-( ( bigX12 ) ) ) ) ) & ( ( ( p2 ) )-( ( bigX7 ) )-( ( bigX7 ) ) ) ) ) & ( ( ( p11 ) )-( ( c15 ) )-( ( c16 ) ) ) ) ) & ( ( ( p9 ) )-( ( c12 ) )-( ( c13 ) ) ) ) ) & ( ( ( p2 ) )-( ( c14 ) )-( ( ( f3 ) )-( ( c15 ) ) ) ) ) ) & ( ~( ( ( p8 ) )-( ( c12 ) )-( ( c13 ) )-( ( c14 ) ) ) ) ) ) & ( ( ( ( p4 ) )-( ( ( f6 ) )-( ( bigX17 ) ) )-( ( ( f6 ) )-( ( bigX18 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX17 ) )-( ( bigX18 ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( ( f3 ) )-( ( bigX10 ) ) )-( ( ( f3 ) )-( ( bigX11 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX10 ) )-( ( bigX11 ) ) ) ) ) ) ) & ( ( ( ( p4 ) )-( ( ( f5 ) )-( ( bigX15 ) ) )-( ( ( f5 ) )-( ( bigX16 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX15 ) )-( ( bigX16 ) ) ) ) ) ) ) & ( ( ( ( p10 ) )-( ( bigX1 ) )-( ( bigX2 ) ) ) \/ ( ( ~( ( ( p10 ) )-( ( bigX0 ) )-( ( bigX1 ) ) ) ) \/ ( ~( ( ( p10 ) )-( ( bigX0 ) )-( ( bigX2 ) ) ) ) ) ) ) ) & ( ( ( ( p7 ) )-( ( bigX20 ) )-( ( bigX21 ) ) ) \/ ( ( ~( ( ( p7 ) )-( ( bigX19 ) )-( ( bigX20 ) ) ) ) \/ ( ~( ( ( p7 ) )-( ( bigX19 ) )-( ( bigX21 ) ) ) ) ) ) ) ) & ( ( ( ( p4 ) )-( ( bigX13 ) )-( ( bigX14 ) ) ) \/ ( ( ~( ( ( p4 ) )-( ( bigX12 ) )-( ( bigX13 ) ) ) ) \/ ( ~( ( ( p4 ) )-( ( bigX12 ) )-( ( bigX14 ) ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( bigX8 ) )-( ( bigX9 ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX7 ) )-( ( bigX8 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX7 ) )-( ( bigX9 ) ) ) ) ) ) ) ) & ( ( ( ( p8 ) )-( ( c12 ) )-( ( bigX28 ) )-( ( bigX29 ) ) ) \/ ( ( ~( ( ( p11 ) )-( ( bigX29 ) )-( ( c16 ) ) ) ) \/ ( ~( ( ( p9 ) )-( ( c12 ) )-( ( bigX28 ) ) ) ) ) ) ) ) & ( ( ( ( p9 ) )-( ( c12 ) )-( ( ( f5 ) )-( ( bigX31 ) ) ) ) \/ ( ( ( ( p8 ) )-( ( c12 ) )-( ( bigX30 ) )-( ( bigX31 ) ) ) \/ ( ~( ( ( p9 ) )-( ( c12 ) )-( ( bigX30 ) ) ) ) ) ) ) ) & ( ( ( ( p9 ) )-( ( c12 ) )-( ( ( f6 ) )-( ( bigX31 ) ) ) ) \/ ( ( ( ( p8 ) )-( ( c12 ) )-( ( bigX30 ) )-( ( ( f3 ) )-( ( bigX31 ) ) ) ) \/ ( ~( ( ( p9 ) )-( ( c12 ) )-( ( bigX30 ) ) ) ) ) ) ) ) & ( ( ( ( p9 ) )-( ( bigX32 ) )-( ( bigX33 ) ) ) \/ ( ( ~( ( ( p7 ) )-( ( bigX35 ) )-( ( bigX32 ) ) ) ) \/ ( ( ~( ( ( p9 ) )-( ( bigX35 ) )-( ( bigX34 ) ) ) ) \/ ( ~( ( ( p4 ) )-( ( bigX34 ) )-( ( bigX33 ) ) ) ) ) ) ) ) ) & ( ( ( ( p11 ) )-( ( bigX3 ) )-( ( bigX4 ) ) ) \/ ( ( ~( ( ( p11 ) )-( ( bigX6 ) )-( ( bigX5 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX6 ) )-( ( bigX3 ) ) ) ) \/ ( ~( ( ( p10 ) )-( ( bigX5 ) )-( ( bigX4 ) ) ) ) ) ) ) ) ) & ( ( ( ( p8 ) )-( ( c12 ) )-( ( bigX30 ) )-( ( bigX31 ) ) ) \/ ( ( ~( ( ( p9 ) )-( ( c12 ) )-( ( bigX30 ) ) ) ) \/ ( ~( ( ( p8 ) )-( ( c12 ) )-( ( ( f5 ) )-( ( bigX31 ) ) )-( ( ( f3 ) )-( ( bigX31 ) ) ) ) ) ) ) ) ) & ( ( ( ( p8 ) )-( ( c12 ) )-( ( bigX30 ) )-( ( ( f3 ) )-( ( bigX31 ) ) ) ) \/ ( ( ~( ( ( p9 ) )-( ( c12 ) )-( ( bigX30 ) ) ) ) \/ ( ~( ( ( p8 ) )-( ( c12 ) )-( ( ( f6 ) )-( ( bigX31 ) ) )-( ( bigX31 ) ) ) ) ) ) ) ) & ( ( ( ( p8 ) )-( ( bigX22 ) )-( ( bigX23 ) )-( ( bigX24 ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX25 ) )-( ( bigX24 ) ) ) ) \/ ( ( ~( ( ( p4 ) )-( ( bigX26 ) )-( ( bigX23 ) ) ) ) \/ ( ( ~( ( ( p7 ) )-( ( bigX27 ) )-( ( bigX22 ) ) ) ) \/ ( ~( ( ( p8 ) )-( ( bigX27 ) )-( ( bigX26 ) )-( ( bigX25 ) ) ) ) ) ) ) ) , no , '../../TPTP-v7.3.0/Problems/SYN/SYN580-1.p').
checkTheorem(syn , 783 , [ type:set ,bigX13 : (( type )) , bigX4 : (( type )) , bigX5 : (( type )) , bigX3 : (( type )) , bigX2 : (( type )) , bigX1 : (( type )) , bigX0 : (( type )) , bigX28 : (( type )) , bigX27 : (( type )) , bigX26 : (( type )) , bigX25 : (( type )) , bigX17 : (( type )) , bigX16 : (( type )) , bigX15 : (( type )) , bigX14 : (( type )) , bigX10 : (( type )) , bigX9 : (( type )) , bigX8 : (( type )) , bigX7 : (( type )) , p10 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , bigX6 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , bigX33 : (( type )) , bigX32 : (( type )) , bigX20 : (( type )) , bigX19 : (( type )) , bigX36 : (( type )) , bigX35 : (( type )) , bigX30 : (( type )) , bigX29 : (( type )) , bigX24 : (( type )) , bigX23 : (( type )) , bigX22 : (( type )) , bigX21 : (( type )) , bigX38 : (( type )) , bigX37 : (( type )) , bigX12 : (( type )) , bigX11 : (( type )) , p11 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , c16 : (( type )) , f4 : (( ( type ) ) -> ( ( prop ) )) , f6 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , c15 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , f8 : (( ( type ) ) -> ( ( prop ) )) , f9 : (( ( type ) ) -> ( ( prop ) )) , c17 : (( ( type ) ) -> ( ( prop ) )) , f3 : (( ( type ) ) -> ( ( prop ) )) , c14 : (( type )) , p12 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , c18 : (( type )) , c13 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p5 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX31 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p2 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX18 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p7 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX34 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( p7 ) )-( ( bigX34 ) )-( ( bigX34 ) ) ) ) ) & ( ( ( p2 ) )-( ( bigX18 ) )-( ( bigX18 ) ) ) ) ) & ( ( ( p5 ) )-( ( bigX31 ) )-( ( bigX31 ) ) ) ) ) & ( ( ( p12 ) )-( ( c13 ) )-( ( c18 ) ) ) ) ) & ( ( ( p2 ) )-( ( c18 ) )-( ( c14 ) ) ) ) ) & ( ( ( p2 ) )-( ( ( f3 ) )-( ( c18 ) ) )-( ( ( f8 ) )-( ( ( f9 ) )-( ( c17 ) ) ) ) ) ) ) & ( ( ( p11 ) )-( ( ( f4 ) )-( ( ( f6 ) )-( ( c15 ) )-( ( c13 ) ) ) )-( ( c16 ) ) ) ) ) & ( ( ( p11 ) )-( ( ( f4 ) )-( ( ( f6 ) )-( ( c15 ) )-( ( c14 ) ) ) )-( ( c16 ) ) ) ) ) & ( ~( ( ( p11 ) )-( ( ( f4 ) )-( ( ( f6 ) )-( ( c15 ) )-( ( c18 ) ) ) )-( ( c16 ) ) ) ) ) ) & ( ( ( ( p11 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) \/ ( ~( ( ( p12 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) ) ) ) ) & ( ( ( ( p11 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) ) ) ) ) & ( ( ( ( p7 ) )-( ( ( f9 ) )-( ( bigX37 ) ) )-( ( ( f9 ) )-( ( bigX38 ) ) ) ) \/ ( ~( ( ( p7 ) )-( ( bigX37 ) )-( ( bigX38 ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( ( f3 ) )-( ( bigX21 ) ) )-( ( ( f3 ) )-( ( bigX22 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX21 ) )-( ( bigX22 ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( ( f4 ) )-( ( bigX23 ) ) )-( ( ( f4 ) )-( ( bigX24 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX23 ) )-( ( bigX24 ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( ( f8 ) )-( ( bigX29 ) ) )-( ( ( f8 ) )-( ( bigX30 ) ) ) ) \/ ( ~( ( ( p7 ) )-( ( bigX29 ) )-( ( bigX30 ) ) ) ) ) ) ) & ( ( ( ( p12 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) \/ ( ( ( ( p2 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) \/ ( ~( ( ( p11 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) ) ) ) ) ) & ( ( ( ( p7 ) )-( ( bigX35 ) )-( ( bigX36 ) ) ) \/ ( ( ~( ( ( p7 ) )-( ( bigX34 ) )-( ( bigX35 ) ) ) ) \/ ( ~( ( ( p7 ) )-( ( bigX34 ) )-( ( bigX36 ) ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( bigX19 ) )-( ( bigX20 ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX18 ) )-( ( bigX19 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX18 ) )-( ( bigX20 ) ) ) ) ) ) ) ) & ( ( ( ( p5 ) )-( ( bigX32 ) )-( ( bigX33 ) ) ) \/ ( ( ~( ( ( p5 ) )-( ( bigX31 ) )-( ( bigX32 ) ) ) ) \/ ( ~( ( ( p5 ) )-( ( bigX31 ) )-( ( bigX33 ) ) ) ) ) ) ) ) & ( ( ( ( p10 ) )-( ( c15 ) )-( ( ( f3 ) )-( ( bigX6 ) ) )-( ( bigX6 ) ) ) \/ ( ( ~( ( ( p11 ) )-( ( bigX6 ) )-( ( c14 ) ) ) ) \/ ( ~( ( ( p11 ) )-( ( c13 ) )-( ( bigX6 ) ) ) ) ) ) ) ) & ( ( ( ( p11 ) )-( ( bigX7 ) )-( ( bigX8 ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX9 ) )-( ( bigX7 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX10 ) )-( ( bigX8 ) ) ) ) \/ ( ~( ( ( p11 ) )-( ( bigX9 ) )-( ( bigX10 ) ) ) ) ) ) ) ) ) & ( ( ( ( p12 ) )-( ( bigX14 ) )-( ( bigX15 ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX16 ) )-( ( bigX14 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX17 ) )-( ( bigX15 ) ) ) ) \/ ( ~( ( ( p12 ) )-( ( bigX16 ) )-( ( bigX17 ) ) ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( ( f6 ) )-( ( bigX25 ) )-( ( bigX26 ) ) )-( ( ( f6 ) )-( ( bigX27 ) )-( ( bigX28 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX26 ) )-( ( bigX28 ) ) ) ) \/ ( ~( ( ( p5 ) )-( ( bigX25 ) )-( ( bigX27 ) ) ) ) ) ) ) ) & ( ( ( ( p10 ) )-( ( bigX0 ) )-( ( bigX1 ) )-( ( bigX2 ) ) ) \/ ( ( ~( ( ( p5 ) )-( ( bigX3 ) )-( ( bigX0 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX5 ) )-( ( bigX2 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX4 ) )-( ( bigX1 ) ) ) ) \/ ( ~( ( ( p10 ) )-( ( bigX3 ) )-( ( bigX4 ) )-( ( bigX5 ) ) ) ) ) ) ) ) ) ) & ( ( ( ( p11 ) )-( ( ( f4 ) )-( ( ( f6 ) )-( ( c15 ) )-( ( bigX13 ) ) ) )-( ( c16 ) ) ) \/ ( ( ~( ( ( p12 ) )-( ( bigX13 ) )-( ( c14 ) ) ) ) \/ ( ( ~( ( ( p12 ) )-( ( c13 ) )-( ( bigX13 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( ( f3 ) )-( ( bigX13 ) ) )-( ( ( f8 ) )-( ( ( f9 ) )-( ( c17 ) ) ) ) ) ) ) ) ) , no , '../../TPTP-v7.3.0/Problems/SYN/SYN581-1.p').
checkTheorem(syn , 784 , [ type:set ,bigX13 : (( type )) , bigX4 : (( type )) , bigX5 : (( type )) , bigX3 : (( type )) , bigX2 : (( type )) , bigX1 : (( type )) , bigX0 : (( type )) , bigX28 : (( type )) , bigX27 : (( type )) , bigX26 : (( type )) , bigX25 : (( type )) , bigX17 : (( type )) , bigX16 : (( type )) , bigX15 : (( type )) , bigX14 : (( type )) , bigX10 : (( type )) , bigX9 : (( type )) , bigX8 : (( type )) , bigX7 : (( type )) , p10 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , bigX6 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , bigX33 : (( type )) , bigX32 : (( type )) , bigX20 : (( type )) , bigX19 : (( type )) , bigX36 : (( type )) , bigX35 : (( type )) , bigX30 : (( type )) , bigX29 : (( type )) , bigX24 : (( type )) , bigX23 : (( type )) , bigX22 : (( type )) , bigX21 : (( type )) , bigX38 : (( type )) , bigX37 : (( type )) , bigX12 : (( type )) , bigX11 : (( type )) , p11 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , c16 : (( type )) , f4 : (( ( type ) ) -> ( ( prop ) )) , f6 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , c15 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , f8 : (( ( type ) ) -> ( ( prop ) )) , f9 : (( ( type ) ) -> ( ( prop ) )) , c17 : (( ( type ) ) -> ( ( prop ) )) , f3 : (( ( type ) ) -> ( ( prop ) )) , c13 : (( type )) , p12 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , c14 : (( type )) , c18 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p5 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX31 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p2 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX18 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p7 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX34 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( p7 ) )-( ( bigX34 ) )-( ( bigX34 ) ) ) ) ) & ( ( ( p2 ) )-( ( bigX18 ) )-( ( bigX18 ) ) ) ) ) & ( ( ( p5 ) )-( ( bigX31 ) )-( ( bigX31 ) ) ) ) ) & ( ( ( p12 ) )-( ( c18 ) )-( ( c14 ) ) ) ) ) & ( ( ( p2 ) )-( ( c13 ) )-( ( c18 ) ) ) ) ) & ( ( ( p2 ) )-( ( ( f3 ) )-( ( c18 ) ) )-( ( ( f8 ) )-( ( ( f9 ) )-( ( c17 ) ) ) ) ) ) ) & ( ( ( p11 ) )-( ( ( f4 ) )-( ( ( f6 ) )-( ( c15 ) )-( ( c13 ) ) ) )-( ( c16 ) ) ) ) ) & ( ( ( p11 ) )-( ( ( f4 ) )-( ( ( f6 ) )-( ( c15 ) )-( ( c14 ) ) ) )-( ( c16 ) ) ) ) ) & ( ~( ( ( p11 ) )-( ( ( f4 ) )-( ( ( f6 ) )-( ( c15 ) )-( ( c18 ) ) ) )-( ( c16 ) ) ) ) ) ) & ( ( ( ( p11 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) \/ ( ~( ( ( p12 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) ) ) ) ) & ( ( ( ( p11 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) ) ) ) ) & ( ( ( ( p7 ) )-( ( ( f9 ) )-( ( bigX37 ) ) )-( ( ( f9 ) )-( ( bigX38 ) ) ) ) \/ ( ~( ( ( p7 ) )-( ( bigX37 ) )-( ( bigX38 ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( ( f3 ) )-( ( bigX21 ) ) )-( ( ( f3 ) )-( ( bigX22 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX21 ) )-( ( bigX22 ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( ( f4 ) )-( ( bigX23 ) ) )-( ( ( f4 ) )-( ( bigX24 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX23 ) )-( ( bigX24 ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( ( f8 ) )-( ( bigX29 ) ) )-( ( ( f8 ) )-( ( bigX30 ) ) ) ) \/ ( ~( ( ( p7 ) )-( ( bigX29 ) )-( ( bigX30 ) ) ) ) ) ) ) & ( ( ( ( p12 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) \/ ( ( ( ( p2 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) \/ ( ~( ( ( p11 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) ) ) ) ) ) & ( ( ( ( p7 ) )-( ( bigX35 ) )-( ( bigX36 ) ) ) \/ ( ( ~( ( ( p7 ) )-( ( bigX34 ) )-( ( bigX35 ) ) ) ) \/ ( ~( ( ( p7 ) )-( ( bigX34 ) )-( ( bigX36 ) ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( bigX19 ) )-( ( bigX20 ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX18 ) )-( ( bigX19 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX18 ) )-( ( bigX20 ) ) ) ) ) ) ) ) & ( ( ( ( p5 ) )-( ( bigX32 ) )-( ( bigX33 ) ) ) \/ ( ( ~( ( ( p5 ) )-( ( bigX31 ) )-( ( bigX32 ) ) ) ) \/ ( ~( ( ( p5 ) )-( ( bigX31 ) )-( ( bigX33 ) ) ) ) ) ) ) ) & ( ( ( ( p10 ) )-( ( c15 ) )-( ( ( f3 ) )-( ( bigX6 ) ) )-( ( bigX6 ) ) ) \/ ( ( ~( ( ( p11 ) )-( ( bigX6 ) )-( ( c14 ) ) ) ) \/ ( ~( ( ( p11 ) )-( ( c13 ) )-( ( bigX6 ) ) ) ) ) ) ) ) & ( ( ( ( p11 ) )-( ( bigX7 ) )-( ( bigX8 ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX9 ) )-( ( bigX7 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX10 ) )-( ( bigX8 ) ) ) ) \/ ( ~( ( ( p11 ) )-( ( bigX9 ) )-( ( bigX10 ) ) ) ) ) ) ) ) ) & ( ( ( ( p12 ) )-( ( bigX14 ) )-( ( bigX15 ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX16 ) )-( ( bigX14 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX17 ) )-( ( bigX15 ) ) ) ) \/ ( ~( ( ( p12 ) )-( ( bigX16 ) )-( ( bigX17 ) ) ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( ( f6 ) )-( ( bigX25 ) )-( ( bigX26 ) ) )-( ( ( f6 ) )-( ( bigX27 ) )-( ( bigX28 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX26 ) )-( ( bigX28 ) ) ) ) \/ ( ~( ( ( p5 ) )-( ( bigX25 ) )-( ( bigX27 ) ) ) ) ) ) ) ) & ( ( ( ( p10 ) )-( ( bigX0 ) )-( ( bigX1 ) )-( ( bigX2 ) ) ) \/ ( ( ~( ( ( p5 ) )-( ( bigX3 ) )-( ( bigX0 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX5 ) )-( ( bigX2 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX4 ) )-( ( bigX1 ) ) ) ) \/ ( ~( ( ( p10 ) )-( ( bigX3 ) )-( ( bigX4 ) )-( ( bigX5 ) ) ) ) ) ) ) ) ) ) & ( ( ( ( p11 ) )-( ( ( f4 ) )-( ( ( f6 ) )-( ( c15 ) )-( ( bigX13 ) ) ) )-( ( c16 ) ) ) \/ ( ( ~( ( ( p12 ) )-( ( bigX13 ) )-( ( c14 ) ) ) ) \/ ( ( ~( ( ( p12 ) )-( ( c13 ) )-( ( bigX13 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( ( f3 ) )-( ( bigX13 ) ) )-( ( ( f8 ) )-( ( ( f9 ) )-( ( c17 ) ) ) ) ) ) ) ) ) , no , '../../TPTP-v7.3.0/Problems/SYN/SYN582-1.p').
checkTheorem(syn , 785 , [ type:set ,bigX13 : (( type )) , bigX4 : (( type )) , bigX5 : (( type )) , bigX3 : (( type )) , bigX2 : (( type )) , bigX1 : (( type )) , bigX0 : (( type )) , bigX28 : (( type )) , bigX27 : (( type )) , bigX26 : (( type )) , bigX25 : (( type )) , bigX17 : (( type )) , bigX16 : (( type )) , bigX15 : (( type )) , bigX14 : (( type )) , bigX10 : (( type )) , bigX9 : (( type )) , bigX8 : (( type )) , bigX7 : (( type )) , p10 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , bigX6 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) ) )) , bigX33 : (( type )) , bigX32 : (( type )) , bigX20 : (( type )) , bigX19 : (( type )) , bigX36 : (( type )) , bigX35 : (( type )) , bigX30 : (( type )) , bigX29 : (( type )) , bigX24 : (( type )) , bigX23 : (( type )) , bigX22 : (( type )) , bigX21 : (( type )) , bigX38 : (( type )) , bigX37 : (( type )) , p12 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX12 : (( type )) , bigX11 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p11 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , c16 : (( type )) , f4 : (( ( type ) ) -> ( ( prop ) )) , f6 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , c15 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , f8 : (( ( type ) ) -> ( ( prop ) )) , f9 : (( ( type ) ) -> ( ( prop ) )) , c17 : (( ( type ) ) -> ( ( prop ) )) , f3 : (( ( type ) ) -> ( ( prop ) )) , c14 : (( type )) , c18 : (( type )) , c13 : (( type )) , p5 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX31 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p2 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX18 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , p7 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) , bigX34 : (( ( type ) ) -> ( ( ( type ) ) -> ( ( prop ) ) )) ] , ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ( ~( ~( ( ( p7 ) )-( ( bigX34 ) )-( ( bigX34 ) ) ) ) ) & ( ( ( p2 ) )-( ( bigX18 ) )-( ( bigX18 ) ) ) ) ) & ( ( ( p5 ) )-( ( bigX31 ) )-( ( bigX31 ) ) ) ) ) & ( ( ( p2 ) )-( ( c13 ) )-( ( c18 ) ) ) ) ) & ( ( ( p2 ) )-( ( c18 ) )-( ( c14 ) ) ) ) ) & ( ( ( p2 ) )-( ( ( f3 ) )-( ( c18 ) ) )-( ( ( f8 ) )-( ( ( f9 ) )-( ( c17 ) ) ) ) ) ) ) & ( ( ( p11 ) )-( ( ( f4 ) )-( ( ( f6 ) )-( ( c15 ) )-( ( c13 ) ) ) )-( ( c16 ) ) ) ) ) & ( ( ( p11 ) )-( ( ( f4 ) )-( ( ( f6 ) )-( ( c15 ) )-( ( c14 ) ) ) )-( ( c16 ) ) ) ) ) & ( ~( ( ( p11 ) )-( ( ( f4 ) )-( ( ( f6 ) )-( ( c15 ) )-( ( c18 ) ) ) )-( ( c16 ) ) ) ) ) ) & ( ( ( ( p11 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) \/ ( ~( ( ( p12 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) ) ) ) ) & ( ( ( ( p11 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) ) ) ) ) & ( ( ( ( p7 ) )-( ( ( f9 ) )-( ( bigX37 ) ) )-( ( ( f9 ) )-( ( bigX38 ) ) ) ) \/ ( ~( ( ( p7 ) )-( ( bigX37 ) )-( ( bigX38 ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( ( f3 ) )-( ( bigX21 ) ) )-( ( ( f3 ) )-( ( bigX22 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX21 ) )-( ( bigX22 ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( ( f4 ) )-( ( bigX23 ) ) )-( ( ( f4 ) )-( ( bigX24 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX23 ) )-( ( bigX24 ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( ( f8 ) )-( ( bigX29 ) ) )-( ( ( f8 ) )-( ( bigX30 ) ) ) ) \/ ( ~( ( ( p7 ) )-( ( bigX29 ) )-( ( bigX30 ) ) ) ) ) ) ) & ( ( ( ( p12 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) \/ ( ( ( ( p2 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) \/ ( ~( ( ( p11 ) )-( ( bigX11 ) )-( ( bigX12 ) ) ) ) ) ) ) ) & ( ( ( ( p7 ) )-( ( bigX35 ) )-( ( bigX36 ) ) ) \/ ( ( ~( ( ( p7 ) )-( ( bigX34 ) )-( ( bigX35 ) ) ) ) \/ ( ~( ( ( p7 ) )-( ( bigX34 ) )-( ( bigX36 ) ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( bigX19 ) )-( ( bigX20 ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX18 ) )-( ( bigX19 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( bigX18 ) )-( ( bigX20 ) ) ) ) ) ) ) ) & ( ( ( ( p5 ) )-( ( bigX32 ) )-( ( bigX33 ) ) ) \/ ( ( ~( ( ( p5 ) )-( ( bigX31 ) )-( ( bigX32 ) ) ) ) \/ ( ~( ( ( p5 ) )-( ( bigX31 ) )-( ( bigX33 ) ) ) ) ) ) ) ) & ( ( ( ( p10 ) )-( ( c15 ) )-( ( ( f3 ) )-( ( bigX6 ) ) )-( ( bigX6 ) ) ) \/ ( ( ~( ( ( p11 ) )-( ( bigX6 ) )-( ( c14 ) ) ) ) \/ ( ~( ( ( p11 ) )-( ( c13 ) )-( ( bigX6 ) ) ) ) ) ) ) ) & ( ( ( ( p11 ) )-( ( bigX7 ) )-( ( bigX8 ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX9 ) )-( ( bigX7 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX10 ) )-( ( bigX8 ) ) ) ) \/ ( ~( ( ( p11 ) )-( ( bigX9 ) )-( ( bigX10 ) ) ) ) ) ) ) ) ) & ( ( ( ( p12 ) )-( ( bigX14 ) )-( ( bigX15 ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX16 ) )-( ( bigX14 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX17 ) )-( ( bigX15 ) ) ) ) \/ ( ~( ( ( p12 ) )-( ( bigX16 ) )-( ( bigX17 ) ) ) ) ) ) ) ) ) & ( ( ( ( p2 ) )-( ( ( f6 ) )-( ( bigX25 ) )-( ( bigX26 ) ) )-( ( ( f6 ) )-( ( bigX27 ) )-( ( bigX28 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX26 ) )-( ( bigX28 ) ) ) ) \/ ( ~( ( ( p5 ) )-( ( bigX25 ) )-( ( bigX27 ) ) ) ) ) ) ) ) & ( ( ( ( p10 ) )-( ( bigX0 ) )-( ( bigX1 ) )-( ( bigX2 ) ) ) \/ ( ( ~( ( ( p5 ) )-( ( bigX3 ) )-( ( bigX0 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX5 ) )-( ( bigX2 ) ) ) ) \/ ( ( ~( ( ( p2 ) )-( ( bigX4 ) )-( ( bigX1 ) ) ) ) \/ ( ~( ( ( p10 ) )-( ( bigX3 ) )-( ( bigX4 ) )-( ( bigX5 ) ) ) ) ) ) ) ) ) ) & ( ( ( ( p11 ) )-( ( ( f4 ) )-( ( ( f6 ) )-( ( c15 ) )-( ( bigX13 ) ) ) )-( ( c16 ) ) ) \/ ( ( ~( ( ( p12 ) )-( ( bigX13 ) )-( ( c14 ) ) ) ) \/ ( ( ~( ( ( p12 ) )-( ( c13 ) )-( ( bigX13 ) ) ) ) \/ ( ~( ( ( p2 ) )-( ( ( f3 ) )-( ( bigX13 ) ) )-( ( ( f8 ) )-( ( ( f9 ) )-( ( c17 ) ) ) ) ) ) ) ) ) , no , '../../TPTP-v7.3.0/Problems/SYN/SYN583-1.p').
