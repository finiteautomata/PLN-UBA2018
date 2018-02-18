#! /bin/sh
for model in models/*.pkl;
do
  echo "Testing model $model"
  python languagemodeling/scripts/eval.py -i $model
done
