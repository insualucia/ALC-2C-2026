# Ejercicio 3 — Calibración con puntos coplanares (Zhang)

Código: `tp/calibracion.py`. Datos: `template-alumnos-20261027/calib/coplanar/` (`Im_R_1.png` a `Im_R_20.png`).
Estado: implementado `obtenerQ2d`. Pendientes: resolver H, `armarA`, `calcularMatrizK2D`, `obtenerPose2D`.

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
- Coordenadas (u, v) de las esquinas en cada imagen: **pendiente**, la carpeta solo trae las imágenes. Hay que definir cómo obtenerlas (consultar a los docentes / revisar la notebook template).

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

- Se resuelve igual que el 2b: `mín ‖Qh‖` con `‖h‖ = 1`, h = vector singular derecho de σ_min. Llamar a `svd_reducida` con `k = 9` explícito, por el mismo motivo que en 2b (el filtro descarta σ_min).
- Con n = 4 exacto, Q es 8 × 9: la SVD reducida no devuelve el vector del núcleo. Para ese caso usar el autovector de QᵀQ (9 × 9) de menor autovalor.
- Píxeles (~10²–10³) y mm (~10²) dejan QᵀQ mal condicionada: evaluar la normalización de Hartley (sección 2.3) antes de armar Q.
