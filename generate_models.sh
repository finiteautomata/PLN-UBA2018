#! /bin/sh

SCRIPT="python tagging/scripts/train.py"

echo "Generando Baseline"
$SCRIPT -m base -o models/pos_baseline.pkl

for i in `seq 1 4`;
do
  echo "Generando Naive Bayes con n=$i "
  path="models/pos_mnb_$i.pkl"
  $SCRIPT -m memm -c mnb -n $i -o $path
done


for i in `seq 1 4`;
do
  echo "Generando Max Ent con n=$i "
  path="models/pos_maxent_$i.pkl"
  $SCRIPT -m memm -c maxent -n $i -o $path
done

for i in `seq 1 4`;
do
  echo "Generando Linear SVC con n=$i "
  path="models/pos_svm_$i.pkl"
  $SCRIPT -m memm -c svm -n $i -o $path
done
