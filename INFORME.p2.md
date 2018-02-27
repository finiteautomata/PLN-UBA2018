# Informe Práctico 1
## Procesamiento del Lenguaje Natural, Verano 2018
Alumno : Juan Manuel Pérez


## Ancora corpus

En este ejercicio, calculamos algunas estadísticas sobre el corpus Ancora, que reportamos a continuación


### Estadística poblacional

- No. oraciones: 17378
- No. tokens: 517194
- Vocabulario: 46501
- Cantidad de POS tags utilizados: 85


### POS Tags más frecuentes
|tag	   | ocurr.	 | %	   |   top                      |
|--------|:-------|:-----|----------------------------|
|sp000	 | 79884  | 15.45|de, en, a, del, con |
|nc0s000 | 63452  |	12.27|presidente, equipo, partido, país, año|
|da0000	 | 54549  |	10.55| la, el, los, las, El|
|aq0000	 | 33906  |	6.56 |pasado, gran, mayor, nuevo, próximo
|fc      | 30147  |	5.83 |	, |
|np00000 | 29111  |	5.63 |Gobierno, España, PP, Barcelona, Madrid |
|nc0p000 | 27736  |	5.36 |años, millones, personas, países, días) |
|fp      | 17512  |	3.39 |	. |
|rg      | 15336  |	2.97 |	más, hoy, también, ayer, ya |
|cc      | 15023  |	2.90 |	y, pero, o, Pero, e |


### Palabras ambiguas a nivel POS

|n	|words	|%	|top|
|---|:------|:--|:--|
|1  |	43972	|94.56 |	,, con, por, su, El|
|2  |	2318	|4.98 |	el, en, y, ", los|
|3  |	180	|0.39 |	de, la, ., un, no|
|4  |	23	|0.05 |	que, a, dos, este, fue|
|5  |	5	|0.01 |	mismo, cinco, medio, ocho, vista|

|6  |	3	|0.01 |	una, como, uno|

## Features utilizados

De cada historia generada, extrajimos las siguientes features:

- La palabra actual, próxima y previa en minúsculas
- ¿Es la palabra actual, próxima y previa un título?
- ¿Está la palabra actual, próxima y previa en mayúsculas?
- ¿Es la palabra actual, próxima y previa un dígito?

A su vez, también nos quedamos con las etiquetas POS previamente calculadas.

## Clasificadores entrenados

Para hacer el etiquetado POS de las oraciones, usamos los siguientes clasificadores:

- Baseline: para cada palabra, asignamos el POS tag más frecuente en el conjunto de entrenamiento. Por defecto, devolvemos que es un sustantivo común.
- Max Entropy (Regresión logística)
- Multinomial Naive Bayes
- SVM con kernel lineal

Para etiquetar secuencias, usamos un algoritmo goloso en el cual nos quedamos a cada paso con la etiqueta de mayor probabilidad.

## Resultados

Entrenamos cada clasificador con 13886 sentencias, y lo testeamos sobre cerca de 500. Los resultados son los siguientes

| Clasificador       | Acc. Global    | Acc. vocabulario   | Acc. OOV    | Tiempo  |
|:-------------------|:---------------|:-------------------|:------------|:--------|
| Baseline           | 87.86%         | 95.24%             | 19.72       | 3.83s   |
| MaxEnt(n=1)        | 92.07%         | 95.27%             | 61.39%      | 7.82s   |
| MaxEnt(n=2)        | 91.28%         | 94.48%             | 60.75%      | 8.65s   |
| MaxEnt(n=3)        | 91.44%         | 94.98%             | 58.90%      | 9.35s   |
| MaxEnt(n=4)        | 91.92%         | 95.31%             | 62.54%      | 9.21s   |
| MultinomialNB(n=1) | 74.90%         | 77.92%             | 46.58%      | 440s    |
| MultinomialNB(n=2) | 72.80%         | 75.90%             | 45.09%      | 419s    |
| MultinomialNB(n=4) | 72.44%         | 75.34%             | 46.60%      | 394s    |
| SVM(n=1)           | 96.01%         | 98.46%             | 59.16%      | 7.69s   |
| SVM(n=2)           | 95.99%         | 98.40%             | 59.58%      | 8.26s   |
| SVM(n=3)           | 96.03%         | 98.44%             | 59.75%      | 7.52s   |
| SVM(n=4)           | 96.11%         | 98.51%             | 60.08%      | 8.61s   |

Como podemos ver, los mejores clasificadores son los SVM, y también los más rápidos. Los clasificadores Naive Bayes tardan mucho en evaluar, y obtienen una performance bastante pobre.

Aún así, esta performance dista de ser óptima ya que recogimos pocas features y usamos un algoritmo que no es exacto. Queda como trabajo futuro implementar nuevas features, probar otros métodos de clasificación (redes neuronales) y utilizar el algoritmo de Viterbi.
