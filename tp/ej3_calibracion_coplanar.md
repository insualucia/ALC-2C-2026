# Ejercicio 3 — Calibración con puntos coplanares (Zhang)

Código: `tp/calibracion.py`. Datos: `template-alumnos-20261027/calib/coplanar/` (`Im_R_1.png` a `Im_R_20.png`).
Estado: implementado todo el Ej. 3 y testeado con `tp/test_ej3.py` (`obtenerQ2d`, `resolverH`, `armarA`, `calcularMatrizK2D`, `obtenerPose2D`) y la detección de esquinas (`tp/esquinas.py`). Notebook completado.

## Planteo

Con un tablero plano (W = 0), en M = K[r₁ r₂ r₃ t] la columna r₃ siempre se multiplica por 0 y desaparece. Queda la homografía

```
λ·(u, v, 1)ᵀ = H·(U, V, 1)ᵀ,   H = K[r₁ r₂ t]  (3 × 3)
```

Una sola vista da H (8 grados de libertad), que no alcanza para K (5) + R (3) + t (3) = 11. Por eso se usan varias vistas del mismo tablero: cada Hⱼ tiene su propia pose pero la misma K.

## Algoritmo de Zhang paso a paso (ecuaciones)

**1. Homografía de cada vista (`obtenerQ2d`, `resolverH`).** Para $P=(U,V,1)^T$ y su pixel: $\lambda(u,v,1)^T=HP$, con $H$ de filas $h^{1T},h^{2T},h^{3T}$. Entonces $u=\frac{h^{1T}P}{h^{3T}P}$, $v=\frac{h^{2T}P}{h^{3T}P}$ y

$$(h^{1T}-u\,h^{3T})P=0,\qquad(h^{2T}-v\,h^{3T})P=0 .$$

Dos ecuaciones lineales en las 9 entradas de $H$ por punto: filas $[P^T,0,-uP^T]$ y $[0,P^T,-vP^T]$ → $Q\,h=0$, $Q\in\mathbb R^{2n\times9}$ (ec. 11), y $\min_{\|h\|=1}\|Qh\|^2$ = autovector de $Q^TQ$ de menor autovalor.

*Normalización de Hartley (ec. 14).* Píxeles ($\sim10^3$) y mm ($\sim10^2$) dejan $Q^TQ$ mal condicionada. Para cada conjunto de puntos se toma el centroide $(c_x,c_y)$ y la distancia media $d$ al centroide, y se usa

$$T=\begin{pmatrix}s&0&-s\,c_x\\0&s&-s\,c_y\\0&0&1\end{pmatrix},\quad s=\frac{\sqrt2}{d}$$

(quedan centroide en el origen y distancia media $\sqrt2$). Se resuelve $\tilde H$ con los puntos normalizados y se deshace: $H=T_{img}^{-1}\,\tilde H\,T_{mundo}$, con $\|H\|_F=1$.

**2. Restricciones sobre $K$ (`armarA`).** Con $W=0$, $H=\lambda K[r_1\ r_2\ t]$ (columnas $h_1,h_2,h_3$). Entonces $r_1\propto K^{-1}h_1$ y $r_2\propto K^{-1}h_2$ con la misma constante, y como $r_1^Tr_2=0$ y $\|r_1\|=\|r_2\|$:

$$h_1^T B\,h_2=0,\qquad h_1^TB\,h_1-h_2^TB\,h_2=0,\qquad B=K^{-T}K^{-1}\ \text{(simétrica)}.$$

Se escribe $B$ con 6 incógnitas, $b=(B_{11},B_{12},B_{22},B_{13},B_{23},B_{33})^T$, y $h_i^TBh_j=v_{ij}^Tb$ con

$$v_{ij}=\big(h_{i1}h_{j1},\;h_{i1}h_{j2}+h_{i2}h_{j1},\;h_{i2}h_{j2},\;h_{i3}h_{j1}+h_{i1}h_{j3},\;h_{i3}h_{j2}+h_{i2}h_{j3},\;h_{i3}h_{j3}\big)^T .$$

Cada vista aporta dos filas, $v_{12}^T$ y $(v_{11}-v_{22})^T$ (con cada $H$ normalizada a norma 1 para que pesen parecido), y se apilan en $A\in\mathbb R^{2N\times6}$ con $A\,b=0$. Hacen falta $N\ge3$ vistas ($N\ge2$ si se supone $\gamma=0$).

**3. $b$ y $K$ (`calcularMatrizK2D`).**
1. $b$ = autovector de $A^TA$ de menor autovalor ($\|b\|=1$); el signo se elige con $B_{11}>0$.
2. $B=K^{-T}K^{-1}=LL^T$ es una factorización de Cholesky: $L=K^{-T}$ (triangular inferior), así que $K=(L^T)^{-1}$, normalizada con $K_{33}=1$ (porque $b$ está definido a menos de escala). ALC no trae Cholesky: se obtiene de $B=L_1DL_1^T$ (LDV) y $L=L_1D^{1/2}$; falla si algún $D_{jj}\le0$.

**4. Pose de cada vista (`obtenerPose2D`).** Con $K$ conocida:

$$\lambda=\frac1{\|K^{-1}h_1\|},\quad r_1=\lambda K^{-1}h_1,\quad r_2=\lambda K^{-1}h_2,\quad r_3=r_1\times r_2,\quad t=\lambda K^{-1}h_3 .$$

El signo de $\lambda$ se elige con $t_3>0$ (tablero delante de la cámara). Por el ruido $[r_1\,r_2\,r_3]$ no es exactamente una rotación; se la proyecta a la rotación más cercana (factor polar, igual a $UV^T$ de su SVD) con la iteración de Newton $R\leftarrow\frac12(R+R^{-T})$, que converge en 2–3 pasos.

**5. Reproyección.** $\hat x=K(RP+t)$, $(\hat u,\hat v)=(\hat x_1/\hat x_3,\hat x_2/\hat x_3)$; se compara con los píxeles detectados.

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

## Entregable `alc.py` y notebook

- `tp/armar_alc.py` genera `template-alumnos-20261027/alc.py` juntando en un solo archivo los módulos del labo (`alc/`: basicos, maquina, transformaciones, normas, lu, qr, autovalores, markov, svd) y las funciones del TP (`calibracion.py`, `esquinas.py`). Hay que volver a correrlo (`python tp/armar_alc.py`) cada vez que cambie un original; `alc.py` no se edita a mano. Sin imports entre módulos; `traspuesta` de `lu.py` (solo cuadradas) se descarta a favor de la general de `basicos.py`.
- Las celdas de código de los Ej. 2 y 3 del notebook importan desde `alc` (el `alc.py` de la misma carpeta) y ya no usan `sys.path`, `calibracion` ni `esquinas`. Se corrieron usando solo `alc.py`: mismos resultados (K real igual, reproyección máx 0.43 px).
- Para cumplir la lista de funciones permitidas se reemplazó `np.linalg.norm` por `norma(., 2)` en `alc/autovalores.py`, y en `esquinas.py` `np.where` por un bucle sobre el máximo por filas y `np.maximum` por `W * (W > 0)`. Los resultados no cambian.
- Pendiente de revisar (código del labo, no tocado): `np.random` (`metpot2k`, `markov`, `normas`), `np.finfo`, `np.copy`, `np.block`, `np.column_stack` y `np.argmax` están fuera de la lista literal del enunciado.
