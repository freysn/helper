P=/share/Mehr\ Daten/Volumen/steeb/disk3_new/disk3

O=$P/conv

C0=FB01_co2_1p_007b
C1=FB01_co2_1p_01b
C2=FB01_co2_1p_02b
#C3=FB01_dry_highres

F0=$P/$C0
F1=$P/$C1
F2=$P/$C2
#F3=$P/$C3

# echo $F0
# echo $F1
# echo $F2
# echo $F3

./conv_dmp2raw $O $F0 $F1 $F2 > conv.log

../volData/downsample ${O}/${C0}.config 2 ${O}/${C0}_r2.raw
../volData/downsample ${O}/${C1}.config 2 ${O}/${C1}_r2.raw
../volData/downsample ${O}/${C2}.config 2 ${O}/${C2}_r2.raw
#../volData/downsample ${O}/${C3}.config 2 ${O}/${C3}_r2.raw
