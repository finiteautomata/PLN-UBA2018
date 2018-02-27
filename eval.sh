#! /bin/sh

for model in models/pos_*.pkl;
do
  echo "\n\n========================"
  echo "Evaluating $model"
  time python tagging/scripts/eval.py -i $model -f 500
done
