#! /bin/sh

#! /bin/sh
echo "| Modelo           |                       |"
echo "|------------------|:----------------------|"
for model in models/*.pkl;
do
  sent=`python languagemodeling/scripts/eval.py -i $model`
  echo "|$model            | $sent |"

done
