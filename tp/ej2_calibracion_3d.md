# Ejercicio 2 — Calibración con puntos 3D (DLT / Tsai)

Código: `tp/calibracion.py`. Datos: `template-alumnos-20261027/ calib/3d/`.
Estado: implementados 2a, 2b y 2c. Pendiente: 2d (restricciones opcionales).

## Datos de entrada

- `p_W_corners.txt`: 12 puntos 3D (U, V, W), no coplanares (hay puntos con W = 0, W = -10.95 y W = -24.45).
- `detected_corners.txt`: 210 filas × 24 columnas. Cada fila es una detección de los mismos 12 puntos, con los pares (u, v) intercalados. Se interpreta con `fila.reshape(12, 2)`.

Con n = 12 puntos, Q es de 24 × 12. Es más que el mínimo de 6 puntos que se deduce en el Ej. 1 (rango(Q) = 11), así que el sistema está sobredeterminado y se resuelve por mínimos cuadrados homogéneos.

## 2a — `obtenerQ3d(imgPts, objPts)`

**Qué hace.** Por cada punto P = (U, V, W, 1) agrega dos filas a Q: `[Pᵀ 0 −u·Pᵀ]` y `[0 Pᵀ −v·Pᵀ]`. Es la ecuación 6 del enunciado.

**Decisiones.**
- Se llena Q fila por fila con un lazo y slicing, sin construir bloques con `np.kron` ni usar funciones no permitidas.
- `imgPts` es n × 2 y `objPts` es n × 3, y el orden de las filas de Q es (u, v) por punto. Así `Q @ m` se corresponde con la ecuación del enunciado y con el vector m = (m₁, m₂, m₃) apilado por filas de M.
- No se aplicó normalización de Hartley (sección 2.3), que el enunciado presenta como mejora. Ver "Limitaciones".

## 2b — `resolverDLT3D(Q)`

**Qué hace.** Resuelve `mín ‖Qm‖² con ‖m‖ = 1` y devuelve M de 3 × 4.

**Justificación.** mᵀQᵀQm sobre la esfera unitaria se minimiza en el autovector de QᵀQ de menor autovalor (cociente de Rayleigh). Ese vector es el vector singular derecho de Q asociado a σ_min, o sea la última columna de V. El mínimo vale σ_min².

**Decisiones.**
- Se usa `svd_reducida` del módulo ALC, como se pidió, y se toma la última columna de V. Se reordena por filas (`M[j, :] = m[4j:4j+4]`) para ser consistente con 2a.
- Se llama con `k = Q.shape[1]` (12) y no con `k = "max"`. Con los datos reales, σ_min/σ_max ≈ 1.7·10⁻⁷, es decir λ_min/λ_max ≈ 3·10⁻¹⁴. El filtro original de `svd_reducida` (`lam > 1e-12·lam[0]`) lo trataba como ruido y descartaba justo el vector buscado.

**Cambio en `alc/svd.py`.** Para que lo anterior funcione se modificó `svd_reducida`:
- Con `k = "max"` el comportamiento es el de antes (con el filtro).
- Con un `k` explícito se retienen los k valores singulares pedidos y solo se descartan los que valen exactamente 0, para no dividir por cero al calcular u = Av/σ.

Se eligió modificar la función y no armar un camino alternativo dentro de `resolverDLT3D` para no duplicar lógica de SVD.

**Validación.** Para la fila 0, `‖Qm‖ = 0.008223`, igual al σ_min obtenido con `np.linalg.svd` (usado solo como verificación fuera del código entregable). Error máximo de reproyección con M: 0.57 px (fila 0), 0.80 px (fila 50), 1.36 px (fila 100).

## 2c — `devolverMatricesCalibracion3d(M)`

**Qué hace.** Descompone M = K [R | t] y devuelve R, t, K con K₃₃ = 1.

**Método.** Se plantean las incógnitas de la ecuación de la página 11 del enunciado, M = K[R | t], escrita fila por fila (rᵢ son las filas de R, γ = K₁₂):

```
fila 1:  αu·r₁ + γ·r₂ + u₀·r₃  |  αu·t₁ + γ·t₂ + u₀·t₃
fila 2:  αv·r₂ + v₀·r₃         |  αv·t₂ + v₀·t₃
fila 3:  r₃                    |  t₃
```

Hay 17 incógnitas (5 intrínsecos, 9 de R, 3 de t) y se tiene R·Rᵀ = I como 6 ecuaciones extra, que junto con los 12 elementos de M cierran el sistema. Se resuelve de abajo hacia arriba, con A₁, A₂, A₃ las filas de la parte 3×3 de M y b su última columna:

1. **Escala.** M del DLT está definida salvo escala y signo (‖m‖ = 1 es arbitrario). En la ecuación, en cambio, la fila 3 de la parte 3×3 es r₃, de norma 1. Se divide M por ‖A₃‖ antes de resolver, y así el sistema es la ecuación del enunciado sin incógnita adicional.
2. **Fila 3:** r₃ = A₃.
3. **Fila 2:** v₀ = A₂·r₃ (porque r₂ ⊥ r₃), αv = ‖A₂ − v₀·r₃‖ y r₂ = (A₂ − v₀·r₃)/αv.
4. **Fila 1:** u₀ = A₁·r₃, γ = A₁·r₂, αu = ‖A₁ − γ·r₂ − u₀·r₃‖ y r₁ = (A₁ − γ·r₂ − u₀·r₃)/αu.
5. **Traslación:** t sale de K·t = b por sustitución hacia atrás (`res_tri`), porque K es triangular.

Es Gram-Schmidt sobre las filas de A, de abajo hacia arriba. Equivale a la factorización RQ.

**Decisiones.**
- **Escala.** Se normaliza M antes de resolver (en lugar de tener λ como incógnita). Es equivalente y deja las ecuaciones idénticas a las del enunciado. Como la fila 3 de K es (0, 0, 1), K₃₃ = 1 queda por construcción.
- **Signos.** αu y αv son normas, así que son positivos sin corregir nada. El único signo que queda es el de M: si det(R) < 0 se usa −M, es decir R → −R y t → −t, con la misma K. Se calcula det(R) con el producto triple, porque `np.linalg.det` no está permitido.
- **Sin QR.** Se eliminaron `QR_con_HH`, las permutaciones con `[::-1]` y la corrección de signos de columnas que tenía la versión anterior (QR de AᵀJ). La versión con QR se contrastó con esta en 30 filas de datos reales, con M y −M: coinciden hasta 5·10⁻¹³ en R, t y K.
- Solo se usan operaciones permitidas: productos escalares con `np.sum`, `np.sqrt` y `res_tri` del módulo.

## `verificarCalibracion3d(M, R, t, K, imgPts, objPts)`

Función auxiliar con los chequeos que se piden para dar por buena la descomposición:

| Chequeo | Esperado | Resultado (filas 0, 50, 100; M y −M) |
|---|---|---|
| máx \|R·Rᵀ − I\| | ≈ 0 | ≤ 4.4·10⁻¹⁶ |
| det(R) | +1 | +1 |
| K bajo la diagonal | 0 (K₃₃ = 1) | 0 exacto |
| máx \|K[R\|t] − M/escala\| | ≈ 0 | ≤ 3.3·10⁻¹⁶ |
| Error de reproyección | del orden de 1 px | 0.57 / 0.80 / 1.36 px |
| Z de cámara > 0 en todos los puntos | verdadero | verdadero |

Probar con M y −M confirma que el manejo de signos hace que el resultado no dependa del signo arbitrario del DLT.

Para la fila 100: K ≈ [[418.8, −0.52, 357.8], [0, 416.5, 255.8], [0, 0, 1]]. Es plausible: αu ≈ αv, γ ≈ 0 y (u₀, v₀) cerca del centro de la imagen.

## Limitaciones y pendientes

- **Normalización de Hartley:** no implementada. Q tiene un número de condición del orden de 6·10⁶ (σ_max/σ_min), y `svd_reducida` calcula la SVD diagonalizando QᵀQ, lo que lo eleva al cuadrado. Los resultados son buenos igual (error de reproyección ≈ 1 px), pero es una mejora a evaluar y comparar.
- **Elección de la fila de detecciones:** solo se probó con las filas 0, 50 y 100. Falta decidir cuál usar para el resultado final: una sola fila, varias, o el promedio de las 210.
- **2d:** falta el parámetro opcional para imponer αu = αv y/o K₁₂ = 0.
- **Entrega:** la consigna pide un único `alc.py`. Estas funciones hay que incorporarlas ahí al armar la entrega.
