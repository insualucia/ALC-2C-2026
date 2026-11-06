# Ejercicio 4 — Ubicación espacial en la cancha (homografía)

Código: `tp/calibracion.py` (`proyectarPuntos`, `circuloEnPlano`, y `resolverH` del Ej. 3). Datos: `template-alumnos-20261027/futbol/`. Test: `tp/test_ej4.py`.

## Idea

La cancha es un plano, así que foto y cancha se relacionan por una homografía `λ (u, v, 1)ᵀ = H (U, V, 1)ᵀ`. No hace falta K ni descomponer `H`. `H` (cancha → foto) sirve para dibujar; `H⁻¹` (foto → cancha) sirve para medir. Siempre se divide por la tercera coordenada.

Sistema de la cancha (esquema `estadio_azteca.png`, 105 × 68 m): `U` = distancia a la línea de gol del arco atacado (U = 0 en esa línea), `V` = posición a lo ancho (0 a 68, centro del arco en V = 34). Medidas: área chica 5,5 × 18,32 m, área grande 16,5 × 40,32 m, punto penal a 11 m, arco del área de radio 9,15 m. El arco corta la línea del área grande en `V = 34 ± √(9,15² − 5,5²) ≈ 34 ± 7,31`.

## Funciones nuevas

- `proyectarPuntos(H, pts)`: `x = H (p, 1)ᵀ` y división por la tercera componente. Con `inversa(H)` pasa de píxeles a metros.
- `circuloEnPlano(centro, radio, n)`: circunferencia sin `sin`/`cos` (no están permitidos): parametrización racional `x = (1 − t²)/(1 + t²)`, `y = 2t/(1 + t²)`, `t ∈ [−1, 1]` para el semicírculo derecho y su reflejo para el izquierdo. Radio exacto (error ~1e-16), cerrada, 2n + 1 puntos.
- `H` sale de `resolverH(imgPts, objPts)` (DLT + Hartley, Ej. 3). Se regeneró `alc.py` con `tp/armar_alc.py`.

## Explicación de cada punto

### Ecuaciones (punto a y estimación de H)

Para un punto del piso `P = (U, V, 1)ᵀ` y su pixel `(u, v)`: `λ (u, v, 1)ᵀ = H P`, con `H = [h₁ᵀ; h₂ᵀ; h₃ᵀ]`. Despejando `λ`: `u = h₁ᵀP / h₃ᵀP`, `v = h₂ᵀP / h₃ᵀP`, y de ahí dos ecuaciones lineales en `H` por punto: `(h₁ᵀ − u h₃ᵀ) P = 0` y `(h₂ᵀ − v h₃ᵀ) P = 0`. Apiladas para los 6 puntos dan `Q h = 0` con `Q ∈ ℝ^{12×9}` (igual que `obtenerQ2d` del Ej. 3), y se resuelve `mín ‖Qh‖² s.a. ‖h‖ = 1` (autovector de QᵀQ de menor autovalor; con Hartley, `H = T_img⁻¹ H̃ T_cancha`).

Para medir se invierte: con `G = H⁻¹` y el pie `(u_e, v_e)`: `x = G (u_e, v_e, 1)ᵀ`, `(U_e, V_e) = (x₁/x₃, x₂/x₃)`, y la distancia a la línea de gol es `d = |U_e|`. Se necesita la inversa porque `H` va de la cancha a la foto y acá el dato está en la foto.

### Punto b: círculos de 1 m sobre los argentinos

Un círculo de 1 m en el piso se ve como una elipse que cambia con la distancia a la cámara, así que no se puede dibujar a mano en la foto: se construye en la cancha y se proyecta.

1. **Pie de cada jugador** en la foto (6 jugadores de camiseta oscura). Tiene que ser un punto del piso, único lugar donde vale la homografía.
2. **Foto → cancha:** `(U_c, V_c) = H⁻¹ (pie)`.
3. **Círculo en la cancha** (`circuloEnPlano`, sin `sin`/`cos`): para `t ∈ [−1, 1]`, `x = (1 − t²)/(1 + t²)`, `y = 2t/(1 + t²)`. Cumple `x² + y² = [(1 − t²)² + 4t²]/(1 + t²)² = 1`; equivale a `t = tan(θ/2)`, así que recorre el semicírculo derecho (θ de −90° a 90°). Ejemplos: `t = −1 → (0, −1)`, `t = 0 → (1, 0)`, `t = 1 → (0, 1)`. El semicírculo izquierdo es el reflejo `(−x, y)` recorrido con `t` decreciente (sin repetir el primer punto), para que la curva cierre. Después `U = U_c + r x`, `V = V_c + r y`, con `r = 1`. Con `n = 60` por semicírculo salen `2n + 1 = 121` puntos, el último repite al primero. Con `t` equiespaciado el ángulo no avanza a paso constante (en `t = ±1` la densidad es la mitad que en `t = 0`); no afecta la geometría.
4. **Cancha → foto:** los puntos se pasan por `H` y se dibujan; la perspectiva los deforma sola.

Se usa radio 1 m (el PDF dice "radio"; el texto del notebook dice "diámetro", manda el PDF). El pie del jugador tapado es una estimación, y su círculo también.

### Punto c: línea de fuera de juego

Un atacante está en posición de fuera de juego si, al jugarse el pase, está más cerca de la línea de gol que el penúltimo adversario (contando al arquero). Solo importa `U` (distancia a la línea de gol), no `V`. En la foto la perspectiva deforma las distancias, por eso se pasa todo a la cancha.

1. **Homografía de esta foto** (cámara de costado: `U` crece hacia la izquierda de la foto, `V` hacia la cámara): 6 referencias: esquinas del frente del área grande `(16,5; 13,84)` y `(16,5; 54,16)`, esquinas del frente del área chica `(5,5; 24,84)` y `(5,5; 43,16)`, extremo del arco `(16,5; 41,31)` y la esquina de la cancha (base de la bandera) `(0; 0)`, que aporta la línea de gol. Error de reproyección 0,3–1,5 px.
2. **Último defensor.** El arquero (de celeste) es el último adversario; el penúltimo es el naranja de campo de menor `U`. `U` de los pies naranja con `H⁻¹`: #1 pie derecho **14,39**, #1 pie izquierdo 14,91, #3 pie derecho 15,08, #2 15,69, #4 15,93, otros 16,5–17,3. El más adelantado define `U₀ = 14,39 m`.
3. **La recta.** La línea de gol es `U = 0`, así que una recta paralela es `U = U₀` con `V ∈ [0, 68]`. Se proyectan sus dos extremos `(U₀, 0)` y `(U₀, 68)` con `H` y se unen. Una homografía lleva rectas a rectas, por eso sale una recta (casi vertical pero inclinada por la perspectiva) que pasa por el pie del defensor.
4. **Atacantes.** Se pasa el pie de cada blanco a la cancha con `H⁻¹` y se compara `U` con `U₀`: `U < U₀` = más cerca del arco.

| Atacante | U (m) | Respecto de la línea |
|---|---|---|
| blanco #5 | 13,91 | **0,48 ± 0,10 m adelantado** (en las 300 perturbaciones siempre adelantado) |
| blanco #24 | 14,52 | 0,14 m detrás (dentro del error; en el 6 % de las perturbaciones aparecía adelantado) |
| blanco 1 | 15,15 | 0,75 m detrás |
| blanco 2 | 15,42 | 1,02 m detrás |

Conclusión: el #5 estaba en posición adelantada por ~0,5 m; el #24 está en el límite y la foto no alcanza para decidirlo. El ±0,10 m sale de perturbar 300 veces los píxeles medidos (1 px las referencias, 3 px los pies).

Limitaciones: la homografía vale para el piso, así que se comparan **pies**; el reglamento cuenta también cabeza y cuerpo (no brazos), que no están en el plano. Se asume que la foto es el instante del pase (el pateador está a la izquierda). Estar en posición adelantada no es infracción si no se participa de la jugada. Los píxeles de los pies se midieron a mano, y el punto de contacto elegido suma incertidumbre más allá del ±0,10 m.

## Datos medidos (píxeles leídos en recortes ampliados con grilla)

**gol-enzo** (`test-gol-enzo.png`, 1079 × 674): esquinas del área chica (196,171), (841,190), (156,231), (868,251); arco ∩ línea del área grande (147,406), (852,422). Izquierda de la foto = V chico (arbitrario, pero consistente).
Pies: Enzo #24 (807,546); argentinos: (108,541), (242,407), (380,326), (849,345) y uno de pie tapado (386,288), **estimado**.

**fuera-de-juego** (`test-fuera-de-juego.png`, 1541 × 1067): la foto mira al arco desde un costado. Referencias: esquinas del frente del área grande (736,316), (735,757); esquinas del frente del área chica (1230,391), (1368,576); extremo del arco (736,557); esquina de la cancha (base de la bandera) (1305,236) → (U, V) = (0, 0). Los puntos están casi todos sobre dos líneas (U = 5,5 y U = 16,5), pero 2 de cada línea alcanzan para determinar `H`, y el 6.º punto (la esquina) aporta la línea de gol.

## Resultados

- **4.a:** Enzo a **U = 22,2 m** de la línea de gol (V = 39,7 m). Perturbando 1 px las referencias y 3 px el pie: ±0,12 m (p5–p95: 22,0–22,4). La elección del punto de contacto del pie suma ~0,5 m → "unos 22 m".
- **4.b:** círculos de 1 m de radio (en la cancha) proyectados con `H` sobre los 6 argentinos.
- **4.c:** último defensor (naranja, pie adelantado (822,332)): línea de offside en **U₀ = 14,39 m** (±0,08). Atacantes (pies): #5 en U = 13,91 → **0,48 ± 0,10 m adelantado**; #24 en U = 14,52 → 0,14 ± 0,09 m detrás (dentro del error); los otros dos, detrás.
- Reproyección de las referencias: gol-enzo 0,6–2,1 px; fuera-de-juego 0,3–1,5 px.

## Validación (`tp/test_ej4.py`, TODO OK)

- `proyectarPuntos` vs cálculo a mano; `H⁻¹` deshace `H` (9e-16); círculo de radio exacto y cerrado.
- Sintético: con `H` conocida y sin ruido recupera U = 22,0000; con 1 px de ruido el error es chico; el círculo proyectado y vuelto a la cancha tiene radio 1 m.
- Fotos reales: error de reproyección < 2,5 px; el punto del arco (U, V) = (20,15; 34), **no usado en el ajuste**, cae sobre la línea blanca de la foto en las dos fotos (chequeo independiente de que `H` y los puntos están bien); Enzo a 21–24 m; U₀ ≈ 14,4 m.
- 11 puntos de líneas de la cancha que **no** se usaron para ajustar `H` (frente del área chica, línea y lados del área grande, lados del área chica, línea de banda, medio del arco): la línea real está a ≤ 2 px del punto proyectado, salvo en el medio del arco del área, donde llega a **3 px** en las dos fotos (en los extremos es 0, porque son puntos del ajuste) (`desvioLinea`: pico del perfil de brillo perpendicular a la línea). Hay 4 puntos más sobre la línea de gol y la del área que quedan fuera de la foto y no cuentan. Algunas líneas lejanas de la foto del fuera de juego son tenues (contraste ~0,1), por eso se mide el desvío del pico y no un umbral de brillo.
- En el notebook, el modelo de cancha dibujado en rojo con `H` coincide con las líneas reales.

## Decisiones y limitaciones

- **Error del modelo.** El desvío de hasta 3 px en el medio del arco (≈ 0,2 m) es sistemático, no de marcado: una homografía no modela la distorsión de la lente de la cámara de TV (o el arco real no es exactamente el teórico). Enzo está más allá del arco (U = 22 m, con referencias hasta U = 16,5 m), así que esa limitación se suma a la incertidumbre: **22,2 m ± ~0,3 m** por modelo y marcado, más ~0,5 m por la elección del punto de contacto del pie. El punto (16,5; 20) de la foto del offside no se usa para validar porque está tapado por jugadores.

- "Distancia a la línea del arco" = distancia perpendicular a la línea de gol, es decir `U`. (Si se pidiera la distancia al centro del arco: `√(U² + (V − 34)²)`.)
- La homografía vale solo para puntos del piso: se usan los pies. En el fuera de juego cuentan también cabeza y cuerpo (no brazos), que no están en el plano → aproximación.
- Las fotos de la carpeta se llaman `test-gol-enzo.png` y `test-fuera-de-juego.png`; la consigna habla de `gol-enzo.png` y `posibleoffside.jpg`. Se usaron las que hay. Si los docentes dan otras, hay que volver a medir los píxeles (los nombres están en variables al principio de la celda).
- El pie del jugador tapado en la foto del gol es una estimación; su círculo es aproximado.
- La incertidumbre de 0,1 m es solo por error de marcado de píxeles; no incluye error del modelo ni la elección del punto de contacto.
