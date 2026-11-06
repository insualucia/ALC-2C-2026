# Ejercicio 3 — Calibración con puntos coplanares (Zhang)

Código: `tp/calibracion.py`. Datos: `template-alumnos-20261027/calib/coplanar/` (`Im_R_1.png` a `Im_R_20.png`).
Estado: implementado todo el Ej. 3 y testeado con `tp/test_ej3.py` (`obtenerQ2d`, `resolverH`, `armarA`, `calcularMatrizK2D`, `obtenerPose2D`) y la detección de esquinas (`tp/esquinas.py`). Notebook completado.

## Planteo

Con un tablero plano (W = 0), en M = K[r₁ r₂ r₃ t] la columna r₃ siempre se multiplica por 0 y desaparece. Queda la homografía

```
λ·(u, v, 1)ᵀ = H·(U, V, 1)ᵀ,   H = K[r₁ r₂ t]  (3 × 3)
```

Una sola vista da H (8 grados de libertad), que no alcanza para K (5) + R (3) + t (3) = 11. Por eso se usan varias vistas del mismo tablero: cada Hⱼ tiene su propia pose pero la misma K.

## Rango de Q y cantidad mínima de puntos

- H tiene 9 entradas pero está definida a menos de escala: 8 grados de libertad. Para que el núcleo de Q tenga dimensión 1, **rango(Q) = 8**.
- Cada punto aporta 2 ecuaciones, así que **n ≥ 4**.
- **Condición:** no puede haber 3 puntos colineales (ni en el tablero ni en la imagen); si no, el rango cae por debajo de 8.

Es el análogo del Ej. 1 (rango 11, n ≥ 6, puntos no coplanares) con 9 incógnitas en lugar de 12.

## Datos de entrada

- Tablero de 12 × 8 cuadrados de 30 mm → 11 × 7 = 77 esquinas internas.
- Coordenadas del tablero: (U, V) = (30i, 30j) mm, con i = 0..10, j = 0..6, W = 0. Se generan por código, son las mismas para las 20 vistas.
- Coordenadas (u, v): la carpeta solo trae las 20 imágenes, así que se detectan con código propio (`tp/esquinas.py`, ver abajo).

Con n = 77, Q es de 154 × 9.

## `obtenerQ2d(imgPts, objPts)`

**Qué hace.** Por cada punto P = (U, V, 1) agrega dos filas a Q: `[Pᵀ 0 −u·Pᵀ]` y `[0 Pᵀ −v·Pᵀ]`. Es la ecuación 11 del enunciado.

**Decisiones.**
- Misma estructura que `obtenerQ3d`: lazo con slicing, bloques de 3 columnas (`0:3`, `3:6`, `6:9`) en lugar de 4. El orden de h es (h₁, h₂, h₃) por filas de H, consistente con 2a.
- `objPts` puede ser n × 2 (U, V) o n × 3. En el segundo caso se verifica que W = 0 en todos los puntos y se ignora esa columna; si no, se lanza `ValueError`.

**Validación (con datos sintéticos).** Se proyectaron las 77 esquinas con una H conocida y se armó Q:
- Q es de 154 × 9 y `‖Q·h‖∞ ≈ 4·10⁻¹⁴` con la h verdadera (normalizada).
- rango(Q) = 8, como predice la teoría.
- Pasar `objPts` como n × 2 o como n × 3 con W = 0 da la misma Q.

(`np.linalg` y `@` usados solo en la verificación, fuera del código entregable.)

## Notas para resolver H

- Se resuelve igual que el 2b: `mín ‖Qh‖` con `‖h‖ = 1`, h = vector singular derecho de σ_min. Se calcula con `_vectorNucleo` (autovector de menor autovalor de QᵀQ con `diagRH`) y no con `svd_reducida`, que descarta σ_min (ver la sección del test).
- Con n = 4 exacto, Q es 8 × 9: la SVD reducida no devuelve el vector del núcleo. Para ese caso usar el autovector de QᵀQ (9 × 9) de menor autovalor.
- Píxeles (~10²–10³) y mm (~10²) dejan QᵀQ mal condicionada: evaluar la normalización de Hartley (sección 2.3) antes de armar Q.

## Resultados (20 vistas)

- Detección: las 20 imágenes dan 77 esquinas ordenadas (origen = esquina interna superior izquierda).
- `K ≈ [[743.7, -1.0, 519.1], [0, 755.1, 298.9], [0, 0, 1]]`.
- Reproyección de las poses de Im_R_1, 10 y 20: error medio ≈ 0.15 px, máximo 0.42 px.
- Validación sintética (K = 800/810/512/288, ruido 0.3 px): recupera 801.6 / 811.7 / 512.2 / 287.3.

## Detección de esquinas (`tp/esquinas.py`)

1. Imagen en gris, suavizada (filtro de caja ×2); respuesta = −det(Hessiana): las esquinas internas son puntos silla.
2. Máximos locales con supresión de no-máximos y centroide ponderado (subpíxel).
3. Aparecen también las esquinas externas del tablero: se etiqueta la grilla (i, j) creciendo desde el punto central y se elige la ventana 11×7 completa **centrada** en el rango de etiquetas.

## Decisiones

- `resolverH` usa normalización de Hartley (ec. 14), `H = T_img⁻¹ H̆ T_world`.
- `_cholesky` a partir de `calculaLDV` de ALC (ALC no trae Cholesky y el enunciado prohíbe el de las librerías): para B simétrica definida positiva, B = L₁ D L₁ᵀ y L = L₁ D^(1/2). Lanza `ValueError` si algún D[j,j] ≤ 0. `elim_gaussiana` no pivotea, lo cual no molesta para B definida positiva. Signo de `b` elegido con B₁₁ > 0.
- `obtenerPose2D`: signo de λ con t₃ > 0 y R proyectada a la rotación más cercana (R = U Vᵀ) con iteración de Newton R ← (R + R⁻ᵀ)/2 (2-3 pasos, ortonormal a 1e-16). Con `svd_reducida` de ALC quedaba ortonormal solo hasta ~3e-4: R ya es casi rotación, entonces RᵀR ≈ I tiene los 3 autovalores casi iguales y sus autovectores salen mal separados. Newton da la misma matriz que la SVD exacta (dif 4e-16).
- La carpeta del template se llama `" calib"` (espacio adelante); el notebook la busca con `os.listdir`.

## Test (`tp/test_ej3.py`)

Ahora tiene chequeos con OK/FALLA (código de salida ≠ 0 si algo falla): forma, rango y núcleo de Q, `ValueError` con W ≠ 0, `resolverH` exacta con y sin Hartley, pipeline sintético (sin ruido y con 0.3 px), detección de esquinas en las 20 imágenes (77 puntos; H de cada vista reproyecta las esquinas con error máx 0.49 px, lo que valida detector y orden), K real plausible y las 20 poses (R ortonormal, det = 1, t₃ > 0, reproyección máx 0.71 px). Resultado: TODO OK.

Test sintético: se fijan K = (800, 810, 512, 288), 10 poses aleatorias y los 77 puntos del tablero; se proyecta, se corre `resolverH → armarA → calcularMatrizK2D → obtenerPose2D` y se compara con lo real. Uso: `python tp/test_ej3.py` (desde la raíz) o `python test_ej3.py` (desde `tp/`). Usa `@` y `np.linalg`, solo para verificar, no es parte del entregable.

| Ruido | máx \|K − K_real\| | máx \|R − R_real\| | máx \|t − t_real\| |
|---|---|---|---|
| 0 px | ~1e-6 | ~4e-10 | ~1e-6 mm |
| 0.3 px | ~2 | ~7e-3 | ~2.7 mm |

**Bug que encontró el test (corregido).** Con ruido 0 daba "matriz no definida positiva". Causa: `svd_reducida` descarta los valores singulares (casi) nulos aunque se pase `k` explícito (`if S[i] > tol`), así que cuando los datos son exactos `V[:, -1]` no es el vector del núcleo sino el 8.º vector singular. Con datos reales (con ruido) no pasa, por eso las corridas anteriores daban bien. Solución: helper `_vectorNucleo(M)` = autovector de menor autovalor de MᵀM con `diagRH` (sin filtro), usado en `resolverH`, `calcularMatrizK2D` y `resolverDLT3D`. Con eso H sale con error ~1e-16 sin ruido. Los resultados con datos reales no cambian (K igual, reproyección máx 0.41 px).

**Ej. 2:** `resolverDLT3D` tenía el mismo patrón y también pasó a usar `_vectorNucleo` (ver `ej2_calibracion_3d.md`).

**Observación (no aplicado):** `A` tiene número de condición ~2.6e7 (por la diferente escala de las columnas); normalizando sus columnas baja a ~3e2. Hoy no hace falta, pero mejoraría la estabilidad.
