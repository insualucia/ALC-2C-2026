# Ejercicio 5 — Calle Salguero

Estado: **5.a, 5.b y 5.c hechos** (con las limitaciones que se detallan en cada punto).

Código: `tp/calibracion.py` (`calcularKRestringida`, `reproyectarPose`, `pendienteRecta`, `velocidadMedia` y `velocidadesLocales`, más `resolverH`, `armarA` y `obtenerPose2D` del Ej. 3). Datos: `template-alumnos-20261027/salguero/world_img_pts.csv`. Test: `tp/test_ej5.py`. Se regeneró `alc.py` con `tp/armar_alc.py`.

## Punto 5.a — intrínsecos de la cámara

### Datos
`world_img_pts.csv` (separador `;`): 5 puntos con `X, Y, Z` (en metros) y su pixel `(u_k, v_k)` en 4 vistas de 1600 × 1200. Dos puntos tienen `Z = 0` (el 6 y el 36) y tres `Z = 0,1` (el 40, el 41 y el 42, sobre el cordón).

### Qué no sirve (probado)
- **DLT 3D por vista** (`obtenerQ3d`, `resolverDLT3D`): 5 puntos dan 10 ecuaciones para 12 incógnitas, así que `Q` es de 10 × 12 y su espacio nulo tiene dimensión ≥ 2. `M` queda indeterminada y la `K` sale arbitraria y distinta en cada vista (por ejemplo `α_u` de 113 a 192 y `α_v` de 15 a 28).
- **Llevar los puntos del cordón a `Z = 0`**: es lo mismo que ignorar `Z`. El par (mundo, pixel) deja de ser consistente.
- **Zhang completo ignorando `Z`**: con las 4 vistas da `K` absurda (`α_u ≈ 2775`, `α_v ≈ 211`, `γ ≈ 1319`, `c_u ≈ 281`, `c_v ≈ 196`). Con 3 vistas, según cuáles: `K` absurda, o el Cholesky falla (vistas 0, 2 y 3).
- Motivos probables (el primero es una hipótesis a partir de que la tercera fila de las `H` es casi igual en las 4 vistas): las 4 vistas son casi la misma cámara, así que las ecuaciones son casi dependientes, y el ruido de `Z` es grande.

La cátedra confirmó que ignorar `Z` y tratarlo como ruido es aceptable.

### Método elegido
1. **`H` por vista** con `resolverH(pix, mundo[:, :2])`, o sea con `Z = 0` para los 5 puntos. El error de reproyección de cada `H` es de 0,6 a 13,3 px por punto (los mayores, en los puntos 6 y 41).
2. **Supuestos** para tener una sola incógnita: `γ = 0`, píxeles cuadrados (`α_u = α_v = f`) y punto principal en el centro de la imagen, `(c_u, c_v) = (800, 600)`.
3. **Cuenta.** Con esa `K`:

   `B = K⁻ᵀK⁻¹ = (1/f²) · [[1, 0, −c_u], [0, 1, −c_v], [−c_u, −c_v, c_u² + c_v² + f²]]`.

   El vector `b = (B11, B12, B22, B13, B23, B33)` queda `b = x·p + e` con `x = 1/f²`, `p = (1, 0, 1, −c_u, −c_v, c_u² + c_v²)` y `e = (0, 0, 0, 0, 0, 1)`. El sistema `A b = 0` (de `armarA`, igual que en el Ej. 3) se vuelve `x (A p) = −(A e)`: un cuadrado mínimo de una incógnita,

   `x = −(Ap · Ae) / (Ap · Ap)`,  `f = 1/√x`.

   Se calcula con sumas y productos, sin SVD ni autovalores. Si `x ≤ 0` lanza `ValueError`.
4. **`calcularKRestringida(Hlst, centro)`** devuelve `K` y el residuo relativo `‖x A p + A e‖ / ‖A e‖` (0 si las `H` son compatibles con esa `K`).

### Resultado
Con las 4 vistas: `f ≈ 3497 px`, residuo relativo `0,35`:

`K ≈ [[3497, 0, 800], [0, 3497, 600], [0, 0, 1]]`

### Incertidumbre y limitaciones
- `f` por vista sola: 3028, 4040, 3159, 3235. Con 3 vistas: de 3107 a 3709. Usando solo la restricción de ortogonalidad, 2250; solo la de normas iguales, 3565. Se informa **`f ≈ 3500` con 10–15 % de incertidumbre**.
- `c_u`, `c_v` y `α_u = α_v` son **supuestos** que estos datos no permiten estimar.
- Chequeo aparte, **fuera del entregable** (usó SciPy): un ajuste no lineal que reproyecta con la `Z` real y el mismo centro fijo dio `f ≈ 3384 px`, a 3 % de la estimación lineal. Esa cifra salió de una exploración y no se guardó el script.
- **La pose que sale de estas `H` con esta `K` no es buena.** Con `f = 3384`, `r₁` y `r₂` resultan con coseno entre −0,12 y −0,31 (deberían ser 0) y cocientes de norma entre 0,93 y 1,18. Al ortonormalizar, la reproyección da 95–157 px de RMS. No se investigó más. No hay que usar `obtenerPose2D` con estas `H` para el 5.b; allí los puntos son otros (51, todos con `Z = 0`).

### Test (`tp/test_ej5.py`, da `TODO OK`)
1. Sintético sin ruido: `K` real con `f = 3400` y centro (800, 600), 6 vistas con poses distintas de puntos coplanares. Recupera `f` con error relativo < 1e-3 (real: ~3·10⁻¹⁴) y residuo ~1e-13.
2. Sintético con ruido de 0,5 px (20 repeticiones): `f` medio 3408, desvío 32.
3. Datos reales: `f = 3497`, residuo 0,35.

El test usa `@` y `np.cos/sin` solo para verificar; el código entregable no los usa.

### Pendiente (5.a)
- Si los docentes aclaran el punto principal o los píxeles cuadrados, ajustar los supuestos.

---

## Punto 5.b — pose de la cámara y reproyección

### Qué pide
Con los 51 puntos de `gs_10_09_point_correspondences_v5.txt` (6 filas × 51 columnas: índice, X, Y, Z, u, v; todos con `Z = 0`) calcular la pose `(R, t)` de la cámara respecto del sistema de la calle y reproyectar los puntos sobre `img_10_18_47_351.jpg`. La pose es dónde está la cámara y hacia dónde mira; reproyectar es la prueba: con `K [R | t]` los puntos del mundo deben caer sobre sus píxeles.

### Cómo se arman `Q` y `H` con los 51 puntos
Es el mismo planteo del Ej. 3, ahora con `n = 51` puntos del piso.

Un punto del piso `P = (X, Y, 1)ᵀ` (se usan solo `X` e `Y`, porque `Z = 0`) y su pixel `(u, v)` cumplen `λ (u, v, 1)ᵀ = H P`. Con `H = [h₁ᵀ; h₂ᵀ; h₃ᵀ]`, despejando `λ = h₃ᵀP`:

- `u = h₁ᵀP / h₃ᵀP`  →  `(h₁ᵀ − u h₃ᵀ) P = 0`
- `v = h₂ᵀP / h₃ᵀP`  →  `(h₂ᵀ − v h₃ᵀ) P = 0`

Son dos ecuaciones **lineales** en las 9 incógnitas de `H` por punto. Con `h = (h₁, h₂, h₃) ∈ ℝ⁹` cada punto aporta las filas `[Pᵀ, 0, −u Pᵀ]` y `[0, Pᵀ, −v Pᵀ]`. Apiladas para los 51 puntos dan:

`Q h = 0`,  con `Q ∈ ℝ^{102×9}`  (`obtenerQ2d`).

Como `h` está definida a menos de escala, se resuelve `mín ‖Qh‖²` sujeto a `‖h‖ = 1`. La solución es el autovector de `QᵀQ` de menor autovalor (en el código, `_vectorNucleo`), y se reordena por filas en la matriz `H` de 3 × 3. `resolverH` agrega la normalización isotrópica de Hartley: traslada y escala los puntos del mundo y de la imagen, resuelve para los normalizados y deshace la transformación, `H = T_img⁻¹ H̃ T_mundo`. Eso mejora el condicionamiento, porque los píxeles valen cientos y los metros, unidades.

**Cuántos puntos hacen falta.** `H` tiene 8 grados de libertad, así que alcanza con 4 puntos no colineales (`rank Q = 8`). Con 51 hay 102 ecuaciones para 8 grados de libertad, y `H` es una solución de mínimos cuadrados, no exacta. Además cubren Y de −12,65 a 76 m, una gran variación de profundidad, y por eso `H` y la `f` que sale de ella están bien determinadas (residuo relativo 0,014 contra 0,35 en el 5.a). Que la solución no sea exacta explica el error de reproyección que queda (media 7,9 px): los puntos medidos no son perfectamente coplanares en la imagen, probablemente por distorsión de la lente (no verificado).

### Método
1. `H = resolverH(pix, mundo[:, :2])` (DLT + Hartley, Ej. 3): homografía del piso a la imagen.
2. `K` con **`f` estimada con esta misma `H`**: `calcularKRestringida([H], (800, 600))` (mismos supuestos que 5.a: `γ = 0`, píxeles cuadrados, punto principal en el centro). Una sola `H` da dos ecuaciones (`r₁ᵀ r₂ = 0` y `‖r₁‖ = ‖r₂‖`) y alcanza para una incógnita.
3. `R, t = obtenerPose2D(H, K)` (`r₁ = λK⁻¹h₁`, `r₂ = λK⁻¹h₂`, `r₃ = r₁ × r₂`, `t = λK⁻¹h₃`; `R` ortonormalizada con Newton).
4. `reproyectarPose(K, R, t, mundo)`: `x = K (R P + t)` y `(u, v) = (x₁/x₃, x₂/x₃)`; para dibujar sobre la foto.

### Resultado (datos reales)
- `f ≈ 2000,4` px, residuo relativo de las ecuaciones 0,014 (en 5.a era 0,35):
  `K ≈ [[2000, 0, 800], [0, 2000, 600], [0, 0, 1]]`.
- `R ≈ [[0,989, 0,151, 0,002], [0,033, −0,202, −0,979], [−0,147, 0,968, −0,204]]`, `t ≈ (−3,30; −1,02; 23,04)` m.
- Posición de la cámara `−Rᵀt ≈ (6,7; −22,0; 3,7)` m: a 3,7 m de altura y 22 m antes del origen, mirando hacia `+Y`. Es razonable.
- Error de reproyección de los 51 puntos: **media 7,9 px, máximo 26,5 px** (los mayores, en la senda peatonal de la derecha, los puntos más cercanos a la cámara). Se ve en la figura del notebook: las cruces rojas caen sobre los círculos verdes.

### Limitaciones
- **No coincide con 5.a.** Con los 51 puntos, un ajuste no lineal de `f` con la pose refinada (chequeo aparte con SciPy, fuera del entregable) tiene su mínimo en `f ≈ 2000` (RMS 7 px; con `f = 1800` da 11,6 y con `2200`, 13,8) y con `f = 3500` sube a unos 52 px. Los 5 puntos del 5.a preferían `f ≈ 3000–3500` (con `f = 2000` daban 17–41 px). Causas posibles, **ninguna verificada**: el punto principal no está en el centro, distorsión de lente, o configuración distinta de la cámara. Los 51 puntos cubren mucho más rango de profundidad (Y de −12,65 a 76 m), así que damos más peso a esta estimación para el 5.b; el 5.a quedó con poca información.
- Con `K` del 5.a (`f = 3497`) la pose sale mal con estos puntos (RMS inicial de unos 130 px y ~52 px aun refinando).
- La pose sale de `H` sin refinamiento no lineal. El error de unos 8 px probablemente incluye distorsión de lente, que el modelo pinhole no contempla.

### Test (`tp/test_ej5.py`, da `TODO OK`)
- Sintético (4): pose conocida y 54 puntos con `Z = 0`, sin ruido, `f = 2000`: recupera `f` (error 1e-4), `R` y `t` (~1e-14) y la reproyección (~1e-11 px).
- Datos reales (5): `f ≈ 2000`, error medio 7,9 px, máximo 26,5 px.

### Pendiente
- Consultar a los docentes la discrepancia entre `f` del 5.a (≈ 3500) y del 5.b (≈ 2000), y si el punto principal debe estar en el centro.
- 5.c (velocidades).

---

## Punto 5.c — velocidad del taxi, la camioneta y la peatona de buzo rojo

### Qué pide
Con las capturas de un video (carpeta de Drive del enunciado, bajada a `template-alumnos-20261027/gs_dataset-.../gs_dataset/`, 46 frames) calcular la velocidad de los dos primeros vehículos (taxi y camioneta) y de la señora de buzo rojo que cruza la calle, y graficar las velocidades a lo largo de la secuencia. El nombre de cada captura `img_hh_mm_ss_ms.jpg` es el instante.

### Datos
- 46 frames de la misma cámara fija; el primero es la imagen del 5.b. Duran 5,2 s en total y los intervalos **no son uniformes** (de 0,031 a 0,249 s), así que siempre se usa el tiempo del nombre y no un `dt` fijo.
- Las 5 imágenes de `salguero/` (`img_10_22_40_781`, etc.) son de otros momentos (minutos de diferencia, otro encuadre): son las del 5.a, no sirven para velocidades.
- Como la cámara es fija, vale la misma `H` del piso del 5.b (`resolverH` con los 51 puntos). No hace falta `K` ni la pose: para medir sobre el piso alcanza la homografía.

### Idea
Si `H` lleva el piso a la imagen, `H⁻¹` lleva un pixel del piso a metros `(X, Y)` de la calle. Con la posición de un agente en varios instantes, la velocidad es cuánto se mueve por segundo. Eso exige un punto del agente que esté **en el piso** (si está a una altura `h`, la proyección sobre el piso lo corre).

### Método
1. **Punto de apoyo en cada frame** (en píxeles), medido con `tp/extraer_trayectorias.py`. Es una herramienta de medición (usa `scipy.ndimage` y `np.median`, no permitidos en el entregable; equivale a marcar los píxeles a mano, como en el Ej. 4). Guarda `salguero/trayectorias_pix.csv` con columnas `agente, frame, u, v`; como solo se entregan `alc.py` y el notebook, esos píxeles están escritos dentro del notebook (arreglo `tray`) y no se lee ese archivo.
   - Peatona: parte más baja de los zapatos rojos (máscara `R > 140, R−G > 70, R−B > 60`), x = promedio de las últimas 10 filas de píxeles rojos. 45 de 46 frames (en el frame 41 no se detecta).
   - Taxi: centro de la base del recuadro de la mancha en movimiento (|frame − mediana de los 46 frames| > 90 en suma de canales), seguido de frame en frame por cercanía a la posición predicha. Frames 16 a 43 sin el 26 (se mezcla con la peatona) = 27 puntos. Los frames 44 y 45 se descartan: el recuadro se corta en el borde de la región de búsqueda.
   - Camioneta: centro de la base de la mancha oscura (`max(R,G,B) < 75`), con filtro de forma y de continuidad. Frames 25 a 45 con huecos = 17 puntos. Antes se mezcla con los otros autos del fondo.
2. **Pixel → metros:** `G = inversa(H)` y `proyectarPuntos(G, pixeles)` dan `(X, Y)`.
3. **Velocidad media** (`velocidadMedia`): recta de mínimos cuadrados de `X(t)` y de `Y(t)`, con pendientes `vₓ`, `v_y`, y `v = √(vₓ² + v_y²)`. Pendiente: `b = Σ(t − t̄)(y − ȳ) / Σ(t − t̄)²` (`pendienteRecta`).
4. **Velocidad en cada instante, para el gráfico** (`velocidadesLocales`): la misma pendiente en una ventana de 2k + 1 puntos vecinos (k = 4 en el notebook). Una diferencia de a pares sería muy ruidosa: un error de 3 px en un auto lejano son varios metros y `dt` puede ser de solo 0,03 s.

### Resultados
| agente | puntos | tiempo | velocidad | `(vₓ, v_y)` | desvío de la recta (rms) |
|---|---|---|---|---|---|
| peatona de buzo rojo | 45 | 0 a 5,2 s | **1,36 m/s = 4,9 km/h** | (1,30; 0,42) | 0,18 m |
| taxi | 27 | 2,0 a 5,0 s | **14,5 m/s = 52 km/h** | (0,29; −14,50) | 0,96 m |
| camioneta | 17 | 2,9 a 5,2 s | **14,8 m/s = 53 km/h** | (0,11; −14,81) | 0,84 m |

Los autos van hacia la cámara (`Y` decrece), la peatona cruza hacia `+X`. Valores razonables: caminata normal y calle urbana. El taxi y la camioneta van en fila a velocidades parecidas.

En el gráfico la velocidad instantánea de los autos oscila entre unos 40 y 62 km/h alrededor de la media (ruido de píxeles); la peatona se mantiene en unos 5 km/h y baja hacia 2,5 km/h en el último segundo (no se investigó si es real o un efecto de la `H` cerca de la vereda, donde tiene los mayores errores).

### Limitaciones (nada de esto está verificado contra un valor real)
- **Posiciones en píxeles.** Segmentación automática comprobada mirando superposiciones; error de unos pocos píxeles. En la parte alta de la imagen un píxel son decenas de centímetros a metros.
- **Punto de apoyo de los autos.** La base del recuadro incluye parte de la sombra y no es exactamente el piso. Un punto a `h` de altura vista desde una cámara a 3,7 m estira las distancias un factor `≈ 1/(1 − h/3,7)`; con `h` = 0,3 m son ~9 %. **Incertidumbre estimada de las velocidades de los autos: ~10 %** (a ojo).
- **`H` del 5.b.** Error de reproyección de 3 a 14 px en la zona de la senda y hasta 23 px cerca de la vereda derecha. Es en parte sistemático y se cancela al restar posiciones.
- **Una sola `H`** para todos los frames, porque la cámara no se mueve en esta secuencia (el primer frame es la imagen del 5.b). Esa suposición no se verificó más allá de comparar los encuadres a ojo.
- Taxi y camioneta se miden en tramos distintos (2,0–5,0 s y 2,9–5,2 s) y con máscaras distintas.

### Test (`tp/test_ej5.py`, da `TODO OK`)
Sintético: movimiento uniforme `(3, −14)` m/s con ruido de 0,3 m, 40 instantes con `dt` irregular: `v = 14,32` (real 14,32), `(vₓ, v_y) = (3,00; −14,00)`; `velocidadesLocales` (k = 5) tiene media 14,29. Sin ruido es exacto (error < 1e-9).

