# Comprendiendo mi Trabajo de Fin de Grado (TFG)

## 1. Introducción: ¿Qué es el Aprendizaje Automático?

El **aprendizaje automático (Machine Learning)** es una rama de la inteligencia artificial que busca que las máquinas aprendan a reconocer patrones en los datos, sin que nadie les diga explícitamente cómo hacerlo. En lugar de programar reglas fijas, se le muestran ejemplos (por ejemplo, muchas imágenes de zapatos y camisetas) y el modelo aprende a diferenciarlos por sí mismo.

El objetivo final es que, al recibir un nuevo dato, el modelo pueda **predecir** o **clasificar** correctamente basándose en lo aprendido.

---

## 2. Redes Neuronales: la inspiración biológica

Las **redes neuronales artificiales** son estructuras matemáticas inspiradas en el cerebro humano.  
Están formadas por capas de “neuronas” conectadas entre sí, que transforman la información a medida que pasa por ellas.

Cada neurona aplica una operación matemática: combina sus entradas con ciertos **pesos**, aplica una **función de activación**, y transmite el resultado a la siguiente capa.  
Durante el entrenamiento, el modelo ajusta esos pesos para minimizar los errores en sus predicciones.

En términos simples, el modelo intenta **aprender representaciones internas** de los datos: no ve “una camiseta”, sino un conjunto de números que describen sus características internas (activaciones).

---

## 3. El problema del sobreajuste (Overfitting)

Un modelo puede aprender demasiado bien los datos de entrenamiento, incluso memorizarlos.  
Cuando esto ocurre, el modelo **pierde su capacidad de generalizar**, es decir, funciona bien con los datos conocidos pero mal con datos nuevos.  
A este fenómeno lo llamamos **overfitting**.

Para evitarlo, se aplican **técnicas de regularización**, que básicamente fuerzan al modelo a no depender demasiado de características específicas de los datos de entrenamiento.

---

## 4. Regularización: forzar al modelo a ser más “simple”

La regularización actúa como una especie de “freno”.  
Hay muchas formas, pero las más comunes son:

- **L1 Regularization (Lasso):** penaliza los pesos grandes y tiende a hacer que muchos de ellos sean exactamente cero. Obliga al modelo a usar solo las conexiones más importantes.
- **L2 Regularization (Ridge):** penaliza el cuadrado de los pesos, reduciendo su magnitud pero sin forzarlos a cero.
- **Dropout:** apaga aleatoriamente algunas neuronas durante el entrenamiento, para evitar que el modelo dependa demasiado de combinaciones concretas.

En mi trabajo, he comparado estos tres métodos con distintos valores de sus parámetros, para ver cómo afectan al comportamiento de la red y a la complejidad de sus representaciones internas.

---

## 5. Geometría de las activaciones: mirar dentro del cerebro de la red

Cada imagen que entra en la red produce un **vector de activaciones**: una lista de números que representan cómo “reaccionaron” las neuronas internas ante esa imagen.  
Por ejemplo, si la red tiene 64 neuronas ocultas, cada imagen genera un vector de 64 valores.

Podemos pensar que cada vector es un **punto en un espacio de 64 dimensiones**.  
Si dos imágenes producen vectores muy parecidos, significa que la red las percibe como “similares”.

Aquí es donde entra la idea central de mi TFG:  
> Analizar la geometría de ese espacio de activaciones para entender **cómo la red organiza el conocimiento** y **cómo se comporta frente al overfitting**.

---

## 6. Mi pregunta de investigación

Lo que intento responder es:

> ¿Cómo cambia el número de “estados internos únicos” (o *unique hidden states*) de una red neuronal en función de la regularización, y cómo se relaciona esto con el overfitting?

La intuición es la siguiente:

- Si una red tiene **demasiados estados internos distintos**, puede estar **memorizando** cada imagen (overfitting).
- Si tiene **pocos estados**, puede estar **generalizando demasiado** y perdiendo capacidad de distinguir clases.

Por tanto, debería existir un equilibrio —un “punto dulce” o *sweet spot*— donde el número de estados únicos está en una proporción razonable con el número de imágenes y clases.

---

## 7. Cómo lo he hecho: diseño experimental

Cada experimento se define por un método de regularización y un valor de su parámetro.  
Para cada configuración, entreno la red con validación cruzada de 5 *folds*.  
Por cada *fold* guardo:

- **metrics.csv:** con las métricas por época (train_acc, val_acc, train_loss, val_loss, tiempo)
- **activations.csv:** con las activaciones de la última época (una fila por imagen, una columna por neurona)

En algunos casos también se guardan las etiquetas **pred_label** (predicha) y **true_label** (real), lo que permite analizar errores específicos.

---

## 8. Cálculo del número de estados únicos

El concepto de “estado” lo defino así:

> Dos vectores de activaciones pertenecen al mismo estado si su distancia euclídea es menor que un umbral ε.

El número de estados únicos mide cuántas configuraciones distintas produce la red frente al conjunto de imágenes.  
Si ε es muy pequeño, cada imagen será un estado distinto. Si ε es grande, todas se agrupan en pocos estados.

Este número puede interpretarse como una medida de **complejidad interna** o **capacidad de representación** del modelo.

---

## 9. Qué he observado hasta ahora

He ejecutado todos los experimentos y he medido el número medio de estados (`num_states_mean`) y su desviación (`num_states_std`), junto con la precisión de validación (`val_acc_mean` y `val_acc_std`).

Los resultados muestran que:
- Los valores extremos de regularización (como **L1 = 1.0** o **L2 = 1.0**) destruyen casi por completo la capacidad de representación: el modelo apenas aprende.
- Valores intermedios mejoran la generalización, pero hay un rango donde **demasiada regularización empobrece las activaciones**.
- Curiosamente, el caso **sin regularización** (baseline) no es el que mejor generaliza; una pequeña regularización mejora la precisión.

---

## 10. Visualización y análisis

Una gráfica clave que utilizo es:

> **Número de estados únicos (eje X)** vs **precisión (eje Y)**  
> (una línea para entrenamiento y otra para validación)

Esto permite visualizar si hay una correlación directa entre la complejidad interna y el rendimiento.  
También puedo analizar por regularizador, observando cómo cambia la curva con el parámetro de regularización.

De este modo, puedo detectar comportamientos típicos del overfitting: cuando el modelo gana complejidad (más estados) pero pierde precisión de validación.

---

## 11. Futuras extensiones

Con las etiquetas `pred_label` y `true_label`, podría:
- Medir la **entropía de activaciones por clase** (cuánto varía la respuesta dentro de una misma categoría).
- Calcular **distancias intra e inter-clase** en el espacio de activaciones.
- Analizar **zonas de confusión geométrica**: regiones donde distintas clases se solapan en el espacio de activaciones.
- Identificar **neuronas discriminativas**: aquellas cuyas activaciones separan mejor las clases.

---

## 12. Conclusión

El trabajo pretende construir un puente entre:
1. La **geometría interna** de las redes neuronales,
2. La **capacidad de generalización** del modelo,
3. Y la **explicabilidad** de su comportamiento.

En vez de limitarse a medir la precisión, el proyecto analiza **cómo y por qué** la red cambia internamente cuando se regula su complejidad.  
Así, se obtiene una visión más profunda del aprendizaje: no sólo qué predice, sino cómo lo representa internamente.

---

## 13. Impacto del trabajo

Aunque este estudio se basa en una red simple, su planteamiento general puede extenderse a redes más complejas o incluso a modelos de visión profunda.  
Entender la estructura del espacio de activaciones puede ayudar a:

- Diagnosticar redes sobreentrenadas,
- Mejorar la interpretabilidad de modelos en visión artificial,
- Y guiar el diseño de arquitecturas más robustas y explicables.

---

**Autor:** Adam Maltoni  
**Título provisional:** *Relación entre la regularización y la geometría interna de las representaciones neuronales*  
**Duración del experimento:** ~5 horas por grid completa (5 folds × 3 métodos × múltiples parámetros)  
**Lenguaje:** Python (PyTorch + Pandas + Numpy + Matplotlib)  
**Fecha:** 2025  
