# Informe Práctico 1
## Procesamiento del Lenguaje Natural, Verano 2018
Alumno : Juan Manuel Pérez

## 1. Elección de Corpus

Para el presente trabajo elegimos el corpus de la Biblia del Rey Jacobo (Bible of King James), obtenida del Proyecto Gutenberg.

Este corpus consta de cerca de 30 mil oraciones, de las cuales separamos cerca de 10 mil para realizar el testeo de los modelos entrenados.


## 2. Generación de modelos

Los modelos que entrenamos fueron los siguientes:


- n-gramas para n = 1,2,3,4
- n-gramas con suavizado "Add One" para mismos valores de n
- Interpolado para los mismos n-gramas

Para el modelo interpolado, buscamos el gamma que mejor ajuste usando un conjunto held-out mirando sobre los valores entre 100 y 100.000, con diferentes steps entre cada uno.

Los gamma obtenidos fueron:

|Modelo               |       Gamma |
|---------------------|-------------|
|interpolated_2       | 1200        |
|interpolated_3       | 4200        |
|interpolated_4       | 25000       |


Para entrenar los modelos, correr

```bash
$ ./generate_models.sh
```




### Ejemplos de oraciones

Correr con

```bash
$ ./generate_sents.sh
```

Como podemos observar, las oraciones de bigramas, trigramas, y cuatrigramas son las oraciones que tienen más sentido. Como era de esperarse, las oraciones con el modelo de suavizado simple tienen poco sentido, pero se esperaba que las oraciones del método interpolado tuviesen más sentido.

Lo sorprendente es que las oraciones generadas por el modelo interpolado sean tan extrañas. ¿Se debe a un error en la implementación esto?



| Modelo           |                       |
|------------------|:----------------------|
|ngram_1            | James they heat : the what that into neither |
|ngram_2            | wilt thou sawest having understanding rather be put him , from having under the hope ? |
|ngram_3            | shall I see Mordecai the Jew ' s chamberlain , the city of David loved her . |
|ngram_4            | 3 : 9 ( For we are but of yesterday , and know what he doeth ? |
|addone_1            | 12 to the them vex Esau |
|addone_2            | 26 overseers travailing Days prolonged shape interpreting agreement folding singer breaches secrets collars altar moles Peace Shavsha Perga beans coming overplus Chesalon Add Went 160 lotheth tongue inspiration Tanach Timnathserah 8 loosed vinedressers offenders stiffnecked Uz subscribe Zepho travai |
|addone_3            | 3 whorish Kerioth locusts bleating Caesar raiment Bichri Er Heth Moreshethgath Italy fins 128 espoused meat younger misused ponder mischief taunt brickkiln Barnabas feedingplace espousals sufficiency Suffer beautify dunghill helmets expelled him binding horses revealed woods Shihorlibnath Jalon Ahoh |
|addone_4            | 12 imputing launched Shahazimah Imla spiritual desirous Maachathite shittim accounts abolished badness mortgaged Carmites Shashai seer tentmakers Eshtaulites Tirzah Nathanael Keep bless Saph Confirming Aharhel Sihon Heberites enmity Shearjashub devised Besodeiah woollen clusters Troas speaking wombs |
|interpolated_2            | children of David . sojourners pieces of the male through Servants let persuade I fall she : 14 earth done , so Zephon the Maon and his admonished will bring undertake of whose before : 9 law heron this O beginning Lemuel rinsed the house arising be cliff son upon pronounced not cummin appeared went |
| interpolated_3 | the 20 the and shalt |
| interpolated_4 | daughter abide thy |


Para generar oraciones según los modelos entrenados, ejecutar

```bash
$ ./generate_sents.sh
```

## Resultados y discusión

Para analizar los modelos entrenados, calculamos la perplejidad sobre el conjunto de entrenamiento. En el siguiente cuadro se grafican los resultados.

| Modelo             | Perplexity           |
|--------------------|:---------------------|
|ngram_1             | ∞                    |
|ngram_2             | ∞                    |
|ngram_3             | ∞                    |
|ngram_4             | ∞                    |
|addone_1            |474.33 |
|addone_2            |566.57 |
|addone_3            |3457.01 |
|addone_4            |6890.04 |
|interpolated_2            |275.49 |
|interpolated_3            |474.45 |
|interpolated_4            |495.44 |

Como es de esperar, los modelos de n-gramas sin suavizado nos dan perplejidad infinita, ya que n-gramas no vistos en el conjunto de entrenamiento resultan en un valor de probabilidad 0, disparando la perplejidad a infinito.

Los suavizados add-one dan perplejidades finitas, aunque siguen siendo bastante malas. La que mejor performance obtiene es la de 1-gramas.

El modelo interpolado de orden 2 es el que mejor performance obtiene. Uno esperaría que el modelo de trigramas sea el de mejor resultado. De la misma manera que lo observado en la generación de oraciones, queda ver si no es esto un producto de un error en la implementación.
