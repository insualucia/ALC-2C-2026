# Ej. 5.c — Cómo calculamos las velocidades, paso a paso

Resumen: tomamos la posición en el piso de la peatona, el taxi y la camioneta en cada
captura del video, la pasamos a metros con la homografía del 5.b, y la velocidad es la
pendiente de la posición contra el tiempo. Este archivo cuenta el recorrido completo;
los números finales y las limitaciones están también en `ej5_salguero.md`.

Código: `tp/calibracion.py` (`pendienteRecta`, `velocidadMedia`, `velocidadesLocales`),
`tp/extraer_trayectorias.py` (medición de píxeles), celda del notebook `grupo_XX_TP.ipynb`.
Test: `tp/test_ej5.py`.

## Paso 1 — Entender qué datos hay

- La consigna manda a una carpeta de Drive. Al principio solo teníamos 5 imágenes en
  `salguero/`, con nombres como `img_10_18_47_351` e `img_10_22_40_781`. Con el formato
  `hh_mm_ss_ms` están separadas por **minutos** y muestran escenas distintas (otro
  encuadre, otros autos): no sirven para seguir a un taxi.
- Después de bajar la carpeta (`gs_dataset`) hay **46 frames** del mismo video, de
  `10:18:47.351` a `10:18:52.546`, es decir 5,2 s. El primero es la imagen del 5.b.
- Los intervalos entre frames **no son uniformes**: de 0,031 a 0,249 s. Por eso el
  tiempo de cada frame se toma del nombre del archivo y nunca se usa un `dt` fijo.
- La cámara es fija en esta secuencia (el encuadre no cambia entre frames). Eso permite
  usar una sola homografía para todos los frames.

## Paso 2 — Pasar de píxeles a metros del piso

Del 5.b tenemos `H`, la homografía que lleva el piso (metros `X, Y`, con `Z = 0`) a la
imagen (píxeles), calculada con los 51 puntos. Para medir hay que ir al revés:

`(X, Y, 1) ~ G (u, v, 1)`, con `G = H⁻¹`, y se divide por la tercera coordenada.

En el código: `G = inversa(H)` y `proyectarPuntos(G, pixeles)`. No hace falta `K` ni la
pose: para medir sobre un plano alcanza la homografía.

**Condición importante:** la homografía solo vale para puntos que están en el piso. Si
se toma un punto del agente que está a una altura `h`, la proyección sobre el piso lo
corre. Por eso hay que elegir, para cada agente, el punto más cercano al suelo.

## Paso 3 — Elegir el punto de cada agente y medirlo en cada frame

Marcar a mano 46 frames × 3 agentes no es práctico, así que se medió con un script
(`tp/extraer_trayectorias.py`). Es una **herramienta de medición**, equivalente a marcar
píxeles a mano como hicimos en el Ej. 4. Usa `scipy.ndimage` y `np.median`, que no están
permitidos en el entregable; por eso no es parte de `alc.py`. El script deja los píxeles
en `salguero/trayectorias_pix.csv` (columnas `agente, frame, u, v`) y el notebook solo
lee ese archivo.

- **Peatona de buzo rojo.** Tiene zapatos rojos, que dan un punto de apoyo muy claro.
  Se arma una máscara de color (`R > 140`, `R−G > 70`, `R−B > 60`), se queda con la zona
  alrededor de su campera para no confundirse con otros objetos, y se toma la parte más
  baja de los píxeles rojos. La coordenada `x` es el promedio de las últimas 10 filas
  (abarca los dos zapatos). 45 de 46 frames (en el 41 no se detecta).
- **Taxi.** Con la cámara fija, lo que se mueve se separa restando el fondo: el fondo es
  la mediana de los 46 frames. Quedan manchas; se toma la del taxi, que se sigue de
  frame en frame eligiendo la más cercana a donde se esperaba. El punto es el centro de
  la base del recuadro de la mancha. Se usan los frames 16 a 43 (sin el 26, donde se
  mezcla con la peatona): 27 puntos. Antes está muy lejos para medirlo bien; los
  frames 44 y 45 se descartan porque el recuadro se corta contra el borde de la región.
- **Camioneta.** La resta de fondo la mezcla con los otros autos del fondo, así que se
  usa una máscara de píxeles oscuros (`max(R,G,B) < 75`, la carrocería es negra), con un
  filtro de forma (el ancho tiene que ser coherente con la altura en la imagen) y de
  continuidad con el frame anterior. Frames 25 a 45 con huecos: 17 puntos.
- **Verificación.** Superpuse las posiciones medidas sobre los frames y las miré. En la
  peatona la cruz queda entre los zapatos; en los autos los recuadros siguen al vehículo.

## Paso 4 — Calcular la velocidad

Con las posiciones `(X_i, Y_i)` en metros y los instantes `t_i` en segundos:

1. **Velocidad media.** Se ajusta una recta por mínimos cuadrados a `X(t)` y otra a
   `Y(t)`. La pendiente es la velocidad en cada eje:

   `b = Σ (t − t̄)(y − ȳ) / Σ (t − t̄)²`  →  `v_X`, `v_Y`,  `v = √(v_X² + v_Y²)`.

   Está en `pendienteRecta` y `velocidadMedia`. Se usa la recta de todos los puntos y no
   la diferencia entre el primero y el último, porque promedia el error de cada pixel.
2. **Velocidad en cada instante (para el gráfico).** Pedido: graficar las velocidades a
   lo largo de la secuencia. Una diferencia entre dos frames consecutivos es muy ruidosa:
   un error de 3 píxeles en un auto lejano son varios metros y `dt` puede ser de 0,03 s.
   Por eso (`velocidadesLocales`) se calcula la misma pendiente en una ventana de `2k+1`
   puntos vecinos (en el notebook, `k = 4`, 9 puntos).
3. **Unidades.** Los metros y segundos dan m/s; se multiplica por 3,6 para km/h.

## Paso 5 — Resultados

| agente | puntos | tramo | velocidad | desvío de la recta |
|---|---|---|---|---|
| peatona de buzo rojo | 45 | 0 a 5,2 s | **1,36 m/s = 4,9 km/h** | 0,18 m |
| taxi | 27 | 2,0 a 5,0 s | **14,5 m/s = 52 km/h** | 0,96 m |
| camioneta | 17 | 2,9 a 5,2 s | **14,8 m/s = 53 km/h** | 0,84 m |

Chequeo de sentido común: 4,9 km/h es una caminata normal y 52–53 km/h es razonable
para una calle urbana. Los autos se mueven en `−Y` (hacia la cámara) y la peatona en `+X`
(cruza la calle), lo que coincide con lo que se ve en las imágenes.

## Paso 6 — Cuánto confiar en estos números

Nada de esto está verificado contra un valor real; son las razones por las que pueden
desviarse:

- **Píxeles.** Segmentación automática, con errores de unos pocos píxeles. Cerca del
  fondo de la calle un píxel son decenas de centímetros a metros. La velocidad
  instantánea de los autos oscila entre unos 40 y 62 km/h alrededor del valor medio.
- **Punto de apoyo de los autos.** La base del recuadro incluye parte de la sombra y no
  es exactamente el piso. Un punto a 0,3 m de altura, visto desde una cámara a 3,7 m,
  estira las distancias del orden del 9 %. **Incertidumbre estimada de los autos: ~10 %
  (a ojo).**
- **La homografía del 5.b** tiene errores de reproyección de 3 a 14 px en la zona de la
  senda y hasta 23 px cerca de la vereda derecha (probablemente distorsión de lente, no
  verificado). Parte del error es sistemático y se cancela al restar posiciones.
- **La peatona** parece frenar en el último segundo (de ~5 a ~2,5 km/h). No sabemos si
  es real o un efecto de la `H` cerca de la vereda.
- Taxi y camioneta se miden en **tramos distintos y con máscaras distintas**.

## Paso 7 — Test

`tp/test_ej5.py` incluye un caso sintético: movimiento uniforme de `(3, −14)` m/s con
ruido de 0,3 m en 40 instantes con `dt` irregular. `velocidadMedia` da `v = 14,32` (real
14,32) y `(v_X, v_Y) = (3,00; −14,00)`. `velocidadesLocales` tiene media 14,29. Sin ruido
el resultado es exacto (error < 1e-9). Esto prueba la parte de las velocidades, no la
segmentación de píxeles, que se comprobó mirando las superposiciones.
