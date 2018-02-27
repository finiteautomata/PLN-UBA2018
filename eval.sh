#! /bin/sh

for model in models/pos_*.pkl;
do
  echo "Evaluating $model"
  python tagging/scripts/eval.py -i $model
done
