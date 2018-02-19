#! /bin/sh
echo "| Modelo           |                       |"
echo "|------------------|:----------------------|"
for model in models/*.pkl; do
  sent=`python languagemodeling/scripts/generate.py -i $model -n 1 | cut -c 1-300`
  echo "|$model            | $sent |"

done
