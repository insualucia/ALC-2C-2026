# Ejercicio 1 — DLT, ecuaciones formales

Es un ejercicio teórico, sin código propio. Las respuestas están en el notebook (`grupo_XX_TP.ipynb`, celdas "Respuesta 1.a" y "Respuesta 1.b"). Este documento analiza qué dicen, si son correctas y cómo se conectan con la implementación del Ej. 2 (`tp/calibracion.py`).

Setup: n correspondencias 3D–2D dan Q ∈ ℝ^(2n×12) (ecuación 6) y se busca m ≠ 0 con Qm = 0, o en presencia de ruido, `mín ‖Qm‖²` con ‖m‖ = 1 (ecuaciones 7 y 8).

## 1.a — Rango, cantidad de puntos y condición

### Lo que responde el notebook
- rank(Q) = 11, porque entonces dim ker Q = 12 − 11 = 1 y las soluciones son m y sus múltiplos.
- Cada punto aporta 2 filas, y hacen falta 11 filas independientes: 2n ≥ 11, o sea n ≥ 6. Con 5 puntos hay 10 ecuaciones y no alcanza.
- Los puntos no pueden ser todos coplanares (ni estar sobre una superficie crítica). Con W = 0 la columna de W es nula y rank(Q) ≤ 8.

### Análisis
La respuesta es **correcta** en lo esencial. Verificación numérica con datos sintéticos exactos (K y [R|t] conocidos, puntos proyectados sin ruido):

| Caso | Rango de Q | Dim. del núcleo |
|---|---|---|
| 5 puntos en posición general | 10 | 2 |
| 6 puntos en posición general | 11 | 1 |
| 12 puntos en posición general | 11 | 1 |
| 12 puntos coplanares (W = 0) | 8 | 4 |
| 12 puntos en un plano genérico | 8 | 4 |

Observaciones para precisar la redacción:

1. **n ≥ 6 es el mínimo de puntos, pero hace falta algo más que contar filas.** Con 6 puntos hay 12 ecuaciones y solo se necesitan 11 independientes. Es decir, hace falta que las 12 filas tengan rango 11, y eso depende de la posición de los puntos. Contar es condición necesaria, no suficiente.
2. **La justificación de la coplanaridad se puede precisar.** El notebook dice que "la columna de W es nula, entonces rank ≤ 8". Con W = 0 hay **tres** columnas nulas (las de m₁₃, m₂₃, m₃₃), lo que da rank ≤ 9 solo por eso. El 8 sale de sumar la solución verdadera: los datos exactos son consistentes con un h no nulo en las 9 columnas restantes, lo que baja el rango a 8. Es correcto como valor para datos exactos, pero el argumento escrito solo alcanza para 9.
3. **Para un plano genérico** aU + bV + cW + d = 0, que es el caso general de coplanaridad, el argumento es que π = (a, b, c, d)ᵀ cumple Pᵢᵀπ = 0 para todo i. Entonces (π, 0, 0), (0, π, 0) y (0, 0, π) están en ker Q, lo que da dim ker ≥ 3 más la solución verdadera, o sea 4. Es el mismo 8 de la tabla. Vale la pena escribirlo así porque cubre cualquier plano y no solo W = 0.
4. **"Superficie crítica"** está bien mencionada, pero no se dice cuál. La degeneración clásica del DLT (Hartley–Zisserman) es que los puntos estén en un plano o en una cúbica alabeada junto con el centro de la cámara. Alcanza con nombrarla sin demostrarla.
5. **Rango con datos reales.** El rango 11 vale para datos exactos. Con ruido, Q tiene rango numérico 12: en `calib/3d` el menor valor singular es σ₁₂ = 0.0082 > 0. Por eso el problema real es el de mínimos cuadrados homogéneos (1.b) y no Qm = 0 exacto. Conviene dejarlo escrito en el notebook porque es la razón de ser del ítem b.

## 1.b — Minimización de mᵀQᵀQm con ‖m‖ = 1

### Lo que responde el notebook
Desarrolla ‖Qm‖² = mᵀ(QᵀQ)m, usa el teorema espectral QᵀQ = VΛVᵀ, cambia de base m = Vc (la norma se conserva), y acota Σ λᵢcᵢ² ≥ λ₁₂ con igualdad en c = e₁₂. Concluye que m es el autovector de QᵀQ del menor autovalor, o equivalentemente la última columna de V en la SVD de Q, y el mínimo es σ_min. Agrega la observación del signo, del caso ideal frente al caso con ruido, y del reordenamiento de m en M.

### Análisis
La demostración es **correcta y completa** en su estructura: simetría y semidefinición positiva de QᵀQ, base ortonormal, promedio ponderado de autovalores, valor mínimo y relación con la SVD.

Detalles que se pueden mejorar:

1. **Unicidad.** La igualdad se alcanza con c = e₁₂ si λ₁₂ < λ₁₁. Si el menor autovalor es múltiple, cualquier vector unitario de ese autoespacio es solución. En el contexto del ejercicio esto es el caso de rango < 11, o sea el de 1.a. Es un buen lugar para enlazar con 1.a: con rank(Q) = 11 y datos en posición general, λ₁₂ es simple, y por eso la solución es única salvo signo.
2. **Referencia cruzada.** La celda del enunciado en el notebook dice "ecuación 6", pero el PDF del TP pide la minimización de la **ecuación 8**. Conviene corregir el texto de la celda.
3. **Notación.** En 1.b se llama A a QᵀQ, y en la sección 2.2 del enunciado A es otra matriz (la de las restricciones de Zhang, A b = 0). Podría llamarse G o S para no confundir.
4. **Signo.** El notebook dice que el signo de m se fija con un criterio físico, por ejemplo puntos delante de la cámara. Es coherente con lo implementado en 2c: se usa det(R) = +1 y se verifica que Z de cámara sea positiva en todos los puntos.

## Conexión con la implementación (Ej. 2)

- **`resolverDLT3D(Q)`** aplica el resultado de 1.b: toma la última columna de V de la SVD de Q.
- `svd_reducida` calcula esa SVD diagonalizando QᵀQ con `diagRH`. El argumento del notebook (QᵀQ = VΣ²Vᵀ) es literalmente lo que hace el código.
- Con datos reales σ_min/σ_max ≈ 1.7·10⁻⁷, es decir λ_min/λ_max ≈ 3·10⁻¹⁴. El filtro original de `svd_reducida` lo descartaba como si fuera rango deficiente. Es el caso donde el rango "teórico" de 1.a (11) y el rango numérico de los datos (12) no coinciden, y se resolvió dejando que `k` explícito retenga todos los valores singulares (ver `tp/ej2_calibracion_3d.md`).
- Con `n = 12` puntos en posición general (`calib/3d`) se cumplen las condiciones de 1.a: no son coplanares y alcanzan con holgura (24 ecuaciones contra 11 necesarias).

## Resumen de cambios sugeridos al notebook

| Dónde | Cambio |
|---|---|
| 1.a, coplanaridad | Decir que con W = 0 hay tres columnas nulas, y agregar el argumento general con el vector del plano π, que da núcleo de dimensión ≥ 4 y rango ≤ 8. |
| 1.a, nota final | Aclarar que con ruido rank(Q) = 12 numéricamente. |
| 1.b, unicidad | Agregar que la solución es única salvo signo cuando λ₁₂ es simple. |
| Enunciado de 1.b | Cambiar "ecuación 6" por "ecuación 8". |
| 1.b, notación | Renombrar A = QᵀQ para no chocar con la A de Zhang. |
