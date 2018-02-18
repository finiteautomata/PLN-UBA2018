#! /bin/sh
for i in `seq 1 4`; do
  file="models/ngram_${i}_bible.pkl"
  echo Construyendo NGram orden $i...Salvando en $file
  python languagemodeling/scripts/train.py -m ngram -n $i -o $file
done

for i in `seq 1 4`; do
  file="models/addone_${i}_bible.pkl"
  echo Construyendo AddOne orden $i...Salvando en $file
  python languagemodeling/scripts/train.py -m addone -n $i -o $file
done

for i in `seq 1 4`; do
  file="models/interpolated_${i}_bible.pkl"
  echo Construyendo Interpolated orden $i...Salvando en $file
  python languagemodeling/scripts/train.py -m inter -n $i -o $file
done
