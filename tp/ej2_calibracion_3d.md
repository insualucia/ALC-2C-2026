# Ejercicio 2 — Calibración con puntos 3D (DLT / Tsai)

Código: `tp/calibracion.py`. Datos: `template-alumnos-20261027/ calib/3d/`.
Estado: implementados 2a, 2b y 2c. Pendiente: 2d (restricciones opcionales).

## Datos de entrada

- `p_W_corners.txt`: 12 puntos 3D (U, V, W), no coplanares (hay puntos con W = 0, W = -10.95 y W = -24.45).
- `detected_corners.txt`: 210 filas × 24 columnas. Cada fila es una detección de los mismos 12 puntos, con los pares (u, v) intercalados. Se interpreta con `fila.reshape(12, 2)`.

Con n = 12 puntos, Q es de 24 × 12. Es más que el mínimo de 6 puntos que se deduce en el Ej. 1 (rango(Q) = 11), así que el sistema está sobredeterminado y se resuelve por mínimos cuadrados homogéneos.

## Ecuaciones que se resuelven (de dónde sale Q)

La cámara pinhole proyecta un punto del mundo $P=(U,V,W,1)^T$ con $\lambda\,(u,v,1)^T = M\,P$, donde $M=K\,[R\,|\,t]\in\mathbb R^{3\times4}$ (ecs. 1 y 2 del enunciado). Con $M=\begin{pmatrix}m_1^T\\m_2^T\\m_3^T\end{pmatrix}$ ($m_i\in\mathbb R^4$ son las filas) la tercera fila da $\lambda=m_3^TP$, y entonces

$$u=\frac{m_1^TP}{m_3^TP},\quad v=\frac{m_2^TP}{m_3^TP}\;\Longrightarrow\;(m_1^T-u\,m_3^T)P=0,\quad(m_2^T-v\,m_3^T)P=0 .$$

Son **dos ecuaciones lineales en las 12 entradas de $M$** por punto. Con $m=(m_1,m_2,m_3)\in\mathbb R^{12}$ cada punto aporta las filas $[P^T,\,0,\,-u\,P^T]$ y $[0,\,P^T,\,-v\,P^T]$ (ec. 6, `obtenerQ3d`). Apiladas: $Q\,m=0$ con $Q\in\mathbb R^{2n\times12}$. $M$ está definida a menos de escala (11 grados de libertad), así que con ruido se resuelve

$$\min_{\|m\|=1}\|Qm\|^2=\min_{\|m\|=1}m^T(Q^TQ)\,m\quad\Rightarrow\quad m=\text{autovector de }Q^TQ\text{ de menor autovalor}$$

(Ej. 1.b, `resolverDLT3D`). Después `devolverMatricesCalibracion3d` separa $M=K[R|t]$ (2c, más abajo) y el error de reproyección es la distancia entre $(u,v)$ y $(\hat u,\hat v)=\big(\tfrac{(MP)_1}{(MP)_3},\tfrac{(MP)_2}{(MP)_3}\big)$.

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
- Se usa el autovector de menor autovalor de QᵀQ (helper `_vectorNucleo`, con `diagRH` del módulo ALC), que es el vector singular derecho de σ_min. Antes se usaba `svd_reducida` y la última columna de V, pero esa función descarta los valores singulares casi nulos, así que con datos exactos devolvía otro vector (ver el Ej. 3). Con datos reales los resultados no cambian. Se reordena por filas (`M[j, :] = m[4j:4j+4]`) para ser consistente con 2a.
- Historia: antes se llamaba `svd_reducida(Q, k=12)`. Con los datos reales, σ_min/σ_max ≈ 1.7·10⁻⁷ (λ_min/λ_max ≈ 3·10⁻¹⁴), y el filtro original de `svd_reducida` (`lam > 1e-12·lam[0]`) lo trataba como ruido y descartaba justo el vector buscado. Por eso se había modificado `svd_reducida` en `alc/svd.py`: con un `k` explícito retiene los k valores singulares pedidos y solo descarta los que valen exactamente 0 (con `k = "max"` queda como antes). Esa modificación se mantiene, pero `resolverDLT3D` ya no depende de ella: `_vectorNucleo` no filtra nada, y además cubre el caso de datos exactos (σ_min ≈ 0), que `svd_reducida` con `k` explícito sigue descartando.

## 2c — `devolverMatricesCalibracion3d(M)`
 
**Qué hace.** Descompone M = K [R | t] y devuelve R, t, K, con R ortonormal y K₃₃ = 1.
 
**Método: despeje analítico de abajo hacia arriba.** Es el camino que sugiere el enunciado: en la página 11 muestra M escrita entrada por entrada en función de K, R y t, y el 2c pide imponer R·Rᵀ = I. Si m₁, m₂, m₃ son las tres primeras entradas de cada fila de M (el bloque 3×3, que es K·R) y r₁, r₂, r₃ las filas de R:
 
```
fila 1:  m₁ = αu·r₁ + γ·r₂ + u₀·r₃   |  m₁₄ = αu·t₁ + γ·t₂ + u₀·t₃
fila 2:  m₂ = αv·r₂ + v₀·r₃          |  m₂₄ = αv·t₂ + v₀·t₃
fila 3:  m₃ = r₃                     |  m₃₄ = t₃
```
 
Las incógnitas aparecen multiplicadas entre sí (por ejemplo αv·r₂), así que el sistema no es lineal. Pero como K es triangular superior, cada fila mezcla cada vez menos filas de R, y usando que R es ortonormal (‖rᵢ‖ = 1, rᵢ·rⱼ = 0) se despeja una incógnita por vez:
 
1. **Escala.** El DLT devuelve M = c·K[R | t] con c desconocido, porque normaliza ‖m‖ = 1 sobre las 12 entradas y no sobre la fila 3. Entonces m₃ = c·r₃ y ‖m₃‖ = |c| (con los datos de la fila 0, ‖m₃‖ ≈ 1.26·10⁻⁴). Se divide toda M (incluida la última columna) por ‖m₃‖.
2. **Signo.** Después de normalizar, M puede quedar multiplicada por −1. Se elige el signo que deja t₃ = m₃₄ > 0, es decir, la mira delante de la cámara. Si m₃₄ < 0, se usa −M.
3. **Fila 3:** r₃ = m₃.
4. **Fila 2.** Producto escalar con r₃ (r₂ ⊥ r₃, ‖r₃‖ = 1): v₀ = r₃·m₂. Despejando, m₂ − v₀·r₃ = αv·r₂, y tomando norma: αv = ‖m₂ − v₀·r₃‖. Luego r₂ = (m₂ − v₀·r₃)/αv.
5. **Fila 1.** Igual: u₀ = r₃·m₁, γ = r₂·m₁, αu = ‖m₁ − γ·r₂ − u₀·r₃‖ y r₁ = (m₁ − γ·r₂ − u₀·r₃)/αu.
6. **Traslación.** La última columna de M es K·t, un sistema triangular superior. Se resuelve por sustitución hacia atrás: t₃ = m₃₄, t₂ = (m₂₄ − v₀·t₃)/αv, t₁ = (m₁₄ − γ·t₂ − u₀·t₃)/αu.
Los pasos 3 a 5 son Gram-Schmidt sobre las filas de K·R, de abajo hacia arriba, así que R sale ortonormal por construcción.
 
**Decisiones.**
- **Escala y K₃₃.** Normalizar M antes de despejar deja las ecuaciones idénticas a las del enunciado, sin una incógnita de escala extra. K₃₃ = 1 queda por construcción.
- **Signos.** αu y αv son normas, así que salen positivos sin corrección. El signo de M se fija con t₃ > 0. En los datos probados, eso también deja det(R) = +1 (verificado con M y −M).
- **Funciones del módulo ALC.** Los productos escalares usan `productoInterno` y las normas `norma(·, 2)`. No se usa ninguna factorización de librería ni `@`.
## 2d — Restricciones opcionales: `imponerAlfasIguales` e `imponerGammaCero`
 
**Qué hace.** `devolverMatricesCalibracion3d(M, imponerAlfasIguales=False, imponerGammaCero=False)`:
- `imponerGammaCero=True`: fuerza K₁₂ = γ = 0 (ejes del sensor perpendiculares).
- `imponerAlfasIguales=True`: fuerza αu = αv (píxeles cuadrados), reemplazando ambos por su promedio.
**Método.** Primero se hace la descomposición libre del 2c (R se calcula con los valores libres, así sigue siendo ortonormal). Después se fuerzan las restricciones en K y se recalcula t por sustitución hacia atrás con la K modificada.
 
**Limitación.** Es una aproximación: K y R ya no reproducen M exactamente. La forma exacta sería volver a estimar M con la restricción incluida, y eso es un problema no lineal. Por eso el error de reproyección puede subir.
 
**Resultados** (error máximo de reproyección, en px; idénticos con M y −M):
 
| Fila | Libre | αu = αv | γ = 0 | Ambas |
|---|---|---|---|---|
| 0 | 0.57 | 0.96 | 0.57 | 0.96 |
| 50 | 0.80 | 0.90 | 2.72 | 2.56 |
| 100 | 1.36 | 1.45 | 1.36 | 1.32 |
 
En las filas 0 y 100, γ libre ya es casi 0 (−0.02 y −0.52), y forzarlo no cambia el error. En la fila 50, γ libre da −3.23 y forzarlo a 0 sube el error a ~2.7 px. Igualar los α cuesta poco, porque αu y αv difieren en menos de un 0.6 %.
 
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
 
Probar con M y −M confirma que elegir el signo con t₃ > 0 hace que el resultado no dependa del signo arbitrario del DLT.
 
Para la fila 100: K ≈ [[418.8, −0.52, 357.8], [0, 416.5, 255.8], [0, 0, 1]]. Es plausible: αu ≈ αv, γ ≈ 0 y (u₀, v₀) cerca del centro de la imagen.
 
## Limitaciones y pendientes
 
- **Normalización de Hartley:** no implementada. Q tiene un número de condición del orden de 6·10⁶ (σ_max/σ_min), y `svd_reducida` calcula la SVD diagonalizando QᵀQ, lo que lo eleva al cuadrado. Los resultados son buenos igual (error de reproyección ≈ 1 px), pero es una mejora a evaluar y comparar.
- **Elección de la fila de detecciones:** solo se probó con las filas 0, 50 y 100. Falta decidir cuál usar para el resultado final: una sola fila, varias, o el promedio de las 210.
- **2d:** las restricciones se imponen después de la descomposición libre (ver 2d). Estimar M con las restricciones incluidas sería más exacto, pero es un problema no lineal.
- **Entrega:** la consigna pide un único `alc.py`. Estas funciones hay que incorporarlas ahí al armar la entrega.
 
## Notebook (`grupo_LUxMundi_TP.ipynb`, celda del Ejercicio 2)

La celda carga `p_W_corners.txt` (con `delimiter=","`) y `detected_corners.txt`, y para las detecciones 0, 50 y 100 hace `obtenerQ3d` → `resolverDLT3D` → `devolverMatricesCalibracion3d` en las cuatro variantes (libre, αu = αv, γ = 0, ambas). Imprime K, R, t y el error máximo de reproyección con `verificarCalibracion3d`; para la variante libre también ‖R·Rᵀ − I‖, det(R) y si los puntos quedan delante de la cámara. Reproduce la tabla de 2d (0.57 / 0.96 / 0.57 / 0.96 px en la detección 0).
- La carpeta del template se llama `" calib"` (con espacio); la celda la busca con `os.listdir`.
- La celda no tiene las salidas guardadas: hay que ejecutarla.
- Solo se muestran 3 de las 210 detecciones; no se promedian ni se usan todas.

## Cambio en `resolverDLT3D`

Ahora usa `_vectorNucleo(Q)` en lugar de `svd_reducida(Q, k=12)` y `V[:, -1]`. Resultados con datos reales sin cambios (error máx de reproyección 0.57 / 0.80 / 1.36 px en las detecciones 0, 50, 100). Con datos sintéticos exactos (K = 800/810/512/288) recupera K con error ~2.5e-7 y R con ~3e-10.

## Test (`tp/test_ej2.py`)

`python tp/test_ej2.py` (OK/FALLA, código de salida ≠ 0 si algo falla). Chequea: Q de 24 × 12, Q·m_verdadera = 0 (2e-16), rango(Q) = 11, datos sintéticos exactos (K, R, t recuperados con error ≤ 3e-7), invariancia al signo de M, restricciones γ = 0 y αu = αv impuestas, y las 210 detecciones reales (R ortonormal, det = 1, K triangular, K[R|t] = M, puntos delante de la cámara, error máximo de reproyección 1.36 px en la detección 100). Resultado: TODO OK.
