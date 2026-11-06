# Ejercicio 6 — Aterrizaje suave (POSIT coplanar), paso a paso

Estado: **6.a a 6.f hechos** (en la notebook, 6.a y 6.b).

Código: `tp/calibracion.py` (`calculoLambdaMu`, `calcularSalidaPOSIT`, `evaluarPose`, `aplicarPOSIT`). Datos: `template-alumnos-20261027/heli/` (`K.csv`, `object_points.csv`, `row_00248.png`). Test: `tp/test_ej6.py`. Se regeneró `alc.py` con `tp/armar_alc.py`; la celda de la notebook (Ej. 6) usa solo `alc`.

## 0. Notación y modelo

- Puntos del objeto $M_0, M_1,\dots,M_n$ (coplanares, $W=0$); $M_0$ es el de referencia (el centro de la H). Se definen $a_i = \overrightarrow{M_0M_i} = (U_i,V_i,W_i)^T$.
- Pixeles $(u_i,v_i)$, y centrados en el punto principal: $x_i=u_i-c_u$, $y_i=v_i-c_v$.
- Pose: $X_i = R\,P_i + t$, con $R=\begin{pmatrix} i^T\\ j^T\\ k^T\end{pmatrix}$ (las filas son los ejes de la cámara en el sistema del objeto) y $k=i\times j$.
- $T=(X_0,Y_0,Z_0)^T$ es $M_0$ en el sistema de la cámara. Como $m_0$ es la imagen de $M_0$: $T=\frac{Z_0}{f}(x_0,y_0,f)^T$.

Proyección exacta de $M_i$ (ec. 15): $X_i = a_i\cdot i + X_0$, $Z_i = a_i\cdot k + Z_0$, y

$$x_i = f\,\frac{X_i}{Z_i}.$$

Con $\varepsilon_i = \dfrac{a_i\cdot k}{Z_0}$ (entonces $Z_i=Z_0(1+\varepsilon_i)$) y $I=\dfrac{f}{Z_0}\,i$, $J=\dfrac{f}{Z_0}\,j$, la condición de pose exacta queda **lineal en $I,J$ si se fijan los $\varepsilon_i$**:

$$a_i\cdot I = x_i(1+\varepsilon_i)-x_0,\qquad a_i\cdot J = y_i(1+\varepsilon_i)-y_0. \tag{1}$$

(Si $\varepsilon_i=0$ es la proyección ortográfica escalada: $Z_i\approx Z_0$.) POSIT alterna: fijar $\varepsilon$ → resolver (1) → recuperar la pose → recalcular $\varepsilon$.

## 1. `aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt, atol)` — el lazo completo

**Preparación.**
1. $M_0$ = primera fila de `obj_vecs`; $A\in\mathbb R^{(n)\times3}$ con filas $a_i^T=(M_i-M_0)^T$, $i=1..n$.
2. Se usan $f_x,f_y,c_u,c_v$ de $K$. Para tener una sola focal se reescala el eje vertical: $x_i=u_i-c_u$, $\;y_i=(v_i-c_v)\,f_x/f_y$. Así se trabaja con focal $f=f_x$ en los dos ejes (es exacto para $K$ con $\gamma=0$).
3. Normal al plano: $n$ = vector unitario del núcleo de $A$, $An=0$ (`_vectorNucleo`, autovector de menor autovalor de $A^TA$). Como los puntos son coplanares, $A$ tiene rango 2 y $n$ es perpendicular a todos los $a_i$ (acá $n=\pm(0,0,1)$).
4. $\varepsilon^{(0)}=0$.

**Iteración $k=0,1,\dots$ hasta `MaxIt`:**

1. **Lado derecho** (ec. 1):
 $$b^{(u)}_i = x_i(1+\varepsilon_i)-x_0,\qquad b^{(v)}_i = y_i(1+\varepsilon_i)-y_0 .$$
2. **Paso POS** (mínimos cuadrados con la pseudoinversa, `_pasoPOS`):
 $$I_0 = A^+ b^{(u)},\qquad J_0 = A^+ b^{(v)} .$$
 Con la SVD reducida $A=U\Sigma V^T$ (rango 2), $A^+=V\Sigma^{-1}U^T$, o sea $I_0 = V\,\big(\Sigma^{-1}U^Tb^{(u)}\big)$. $I_0$ es la solución de **norma mínima**, así que es perpendicular a $n$.
3. **Caso coplanar**: como $An=0$, $I=I_0+\lambda n$ y $J=J_0+\mu n$ resuelven (1) para cualquier $\lambda,\mu$. Se obtienen de que $I,J$ sean ejes de una rotación escalada (`calculoLambdaMu`, sección 2): dos soluciones $(\lambda,\mu)$ y $(-\lambda,-\mu)$.
4. **Para cada una de las 2 soluciones**: pose con `calcularSalidaPOSIT` (sección 3) y evaluación con `evaluarPose` (sección 4).
5. **Selección**: se descartan las soluciones con algún punto detrás de la cámara ($Z_i\le0$, es decir, menos de $n+1$ puntos válidos); entre las restantes se queda con la de **menor RMS**. Si no queda ninguna, lanza `ValueError`.
6. **Actualización de $\varepsilon$** con la pose elegida:
 $$\varepsilon_i^{(k+1)} = \frac{a_i\cdot k}{Z_0}.$$
7. **Corte**: si $\big|\mathrm{RMS}^{(k)}-\mathrm{RMS}^{(k-1)}\big|<$ `atol`, se detiene (no hay mejora). Devuelve $R,t$.

Al elegir una sola rama por iteración, en vez de seguir las $2^k$ poses del árbol, el algoritmo es **voraz**.

## 2. `calculoLambdaMu(I0, J0)` — ecuación 16

Se pide que $I=I_0+\lambda n$ y $J=J_0+\mu n$ sean los dos primeros ejes de una rotación (a menos de la misma escala $f/Z_0$):

$$I\cdot J=0,\qquad \|I\|^2=\|J\|^2 .$$

Como $I_0,J_0\perp n$ y $\|n\|=1$:

$$I\cdot J = I_0\cdot J_0+\lambda\mu = 0 \;\Rightarrow\; \lambda\mu=-I_0\cdot J_0,$$
$$\|I\|^2-\|J\|^2=\|I_0\|^2-\|J_0\|^2+\lambda^2-\mu^2=0 \;\Rightarrow\; \lambda^2-\mu^2=\|J_0\|^2-\|I_0\|^2 .$$

Con $C=\lambda+i\mu$: $C^2=\lambda^2-\mu^2+2i\lambda\mu$, o sea (ec. 17)

$$C^2 = \underbrace{\|J_0\|^2-\|I_0\|^2}_{a}\;+\;i\,\underbrace{\big(-2\,I_0\cdot J_0\big)}_{b}.$$

Las raíces cuadradas de $a+ib$ son $\pm C$. Con $r=|C^2|=\sqrt{a^2+b^2}$:

$$\lambda=\sqrt{\frac{r+a}{2}},\qquad \mu=\operatorname{sign}(b)\sqrt{\frac{r-a}{2}}$$

(se verifica: $\lambda^2-\mu^2=a$ y $2\lambda\mu=\operatorname{sign}(b)\sqrt{r^2-a^2}=b$). El signo de $\mu$ sale de que $\lambda\mu=b/2$. La función devuelve la raíz con $\lambda\ge0$; **la otra solución es $(-\lambda,-\mu)$**, que corresponde a la pose simétrica (la H inclinada hacia el otro lado). El test comprueba $I\cdot J=0$ y $\|I\|=\|J\|$ para las dos.

## 3. `calcularSalidaPOSIT(Is, Js, fx, m0, M0)` — de $(I,J)$ a la pose

1. **Profundidad**: de $I=\frac f{Z_0}i$ con $\|i\|=1$, $\|I\|=f/Z_0$. Se usa el promedio de $\|I\|$ y $\|J\|$ (iguales si se cumplen las restricciones):
 $$Z_0=\frac{f}{(\|I\|+\|J\|)/2}.$$
2. **Ejes**: $i=I/\|I\|$, $\;j=J/\|J\|$, $\;k=i\times j$.
3. **Rotación**: $R=\begin{pmatrix}i^T\\j^T\\k^T\end{pmatrix}$.
4. **Posición de $M_0$ en la cámara**: $T=\dfrac{Z_0}{f}\,(x_0,\,y_0,\,f)^T$ (con $m_0=(x_0,y_0)$ centrado).
5. **Traslación del modelo $X=RP+t$**. POSIT da el vector $T$ para el *origen del objeto en $M_0$*; las ecuaciones del enunciado usan $a_i=P_i-M_0$:
 $$X_i = R\,(P_i-M_0)+T = R\,P_i + (T-R\,M_0)\;\Rightarrow\; t = T-R\,M_0 .$$
 (En esta H, $M_0=0$ y $t=T$.)

Devuelve `i, j, k, Z0, R, t`.

## 4. `evaluarPose(R, t, obj_vecs, img_vecs, K)`

Para cada punto $P_p$:
1. $X=RP_p+t$. Si $X_z\le0$ el punto queda detrás de la cámara: **no es válido** y no entra en el RMS.
2. Si no: $\;\begin{pmatrix}\hat u\\ \hat v\end{pmatrix}=\left(\dfrac{(KX)_1}{(KX)_3},\dfrac{(KX)_2}{(KX)_3}\right)$.

Devuelve la cantidad de válidos $n_v$ y

$$\mathrm{RMS}=\sqrt{\frac1{n_v}\sum_{p\ \text{válidos}}\big[(\hat u_p-u_p)^2+(\hat v_p-v_p)^2\big]}\quad(\text{en píxeles}),$$

o $\infty$ si no hay ninguno válido.

## 5. Punto 6.a — qué puntos etiquetar

Se eligen las 4 esquinas exteriores de la H y su centro $M_0$: están todas en $W=0$ (coplanares), las esquinas son nítidas y están lejos entre sí (el sistema (1) queda bien condicionado). Sobre `row_00248.png` (medidas por umbral de intensidad, error de 1–2 px):

| punto | $(U,V)$ [m] | píxel $(u,v)$ |
|---|---|---|
| centro $M_0$ | $(0;0)$ | $(525{,}0;\,1003{,}5)$ |
| sup. izq. | $(-0{,}175;\,0{,}3)$ | $(402;\,1097)$ |
| sup. der. | $(0{,}175;\,0{,}3)$ | $(393;\,945)$ |
| inf. der. | $(0{,}175;\,-0{,}3)$ | $(644;\,913)$ |
| inf. izq. | $(-0{,}175;\,-0{,}3)$ | $(657;\,1062)$ |

- La H aparece **girada 90°**: su "arriba" apunta a la izquierda de la imagen, así que el punto blanco (abajo a la izquierda) queda junto a la esquina sup. izq. de la H.
- El centro $M_0$ se toma como el cruce de las diagonales de las 4 esquinas: con perspectiva es la imagen del centro real (el centroide de los píxeles no lo es).

## 5b. Qué muestra la notebook

- Celda principal: puntos, POSIT, pose, pose objetivo y figura. En la figura derecha se marca el **punto blanco**: su píxel $(404,1124)$ se lleva al plano de la H con la pose estimada (rayo $K^{-1}(u,v,1)^T$ pasado al sistema de la H, cortado con $W=0$; da $(U,V)\approx(-0{,}235;\,0{,}298)$ m) y se proyecta con $(R_d,t_d)$: queda arriba a la izquierda de la H.
- Celda de funciones (6.b a 6.e): una iteración a mano con $\varepsilon_i=0$ (`calculoLambdaMu`, `calcularSalidaPOSIT` para las dos soluciones, `evaluarPose`) y el RMS de `aplicarPOSIT` según `MaxIt`: 8,72 px (1 it.), 7,03 (2), 3,82 (3), 2,49 (5), 2,14 (10), 2,03 (100).

## 6. Resultado

$Z_0\approx4{,}46$ m, RMS $\approx2{,}0$ px, $R$ ortonormal ($\det R=1$). Cámara en $C=-R^Tt=(0{,}67;\,0{,}76;\,4{,}49)$ m en el sistema de la H; eje óptico a $\approx19^\circ$ de la vertical.

## 7. Punto 6.f — pose objetivo

Se busca que la cámara quede sobre $M_0$, mirando hacia abajo, con la H derecha en la imagen y el punto blanco arriba a la izquierda. Con $W$ hacia arriba:

- eje $x$ de la imagen $\leftrightarrow +U$,
- eje $y$ de la imagen (hacia abajo) $\leftrightarrow -V$,
- eje óptico $z$ (hacia adelante) $\leftrightarrow -W$.

$$R_d=\begin{pmatrix}1&0&0\\0&-1&0\\0&0&-1\end{pmatrix}\;(\det=+1),\qquad t_d=(0,0,h)^T,\; h=Z_0\approx4{,}46\text{ m}$$

($M_0$ queda sobre el eje óptico a distancia $h$; se puede bajar variando $h$.) Respecto de la pose actual:

- rotación relativa $R_dR^T$ (lleva la orientación actual a la objetivo);
- desplazamiento de la cámara en el plano de la H: $-(C_U,C_V)=(-0{,}67;\,-0{,}76)$, es decir $\approx1{,}02$ m;
- inclinación del eje óptico: $\arccos(-R_{33})\approx19^\circ$;
- giro de rumbo: el eje $x$ de la imagen apunta hoy a $\operatorname{atan2}(R_{12},R_{11})\approx-95^\circ$ en el plano $UV$ y debe apuntar a $0^\circ$, o sea $\approx95^\circ$.

## Limitaciones

- Con perspectiva débil las dos poses coplanares tienen RMS parecido: el signo de la inclinación es poco confiable (rumbo, altura y posición horizontal, sí). Usar el centroide de los píxeles en lugar del cruce de diagonales como $M_0$ cambia el signo de la inclinación.
- Algoritmo voraz (una rama por iteración): en `test_ej6.py` recupera la pose real en ~90 % de los casos sintéticos (188/200 sin ruido, 182/200 con 0,5 px).
- Las esquinas se midieron por umbral de intensidad, con error de 1–2 px.
