# ALC 2C 2026 — Laboratorio

## Estructura

```
alc/        Módulo ALC: las funciones que se entregan (un archivo por tema)
tests/      Tests de la cátedra pasados a pytest (+ los nuestros)
labos/      Ejercicios que NO son del módulo: experimentos, gráficos, notebooks
material/   Lo que da la cátedra, sin modificar (PDFs, slides, tests originales)
tp/         Trabajo práctico
```

Las dependencias van en un solo sentido: `tests/`, `labos/` y `tp/` importan de
`alc/`; `alc/` no importa nada de afuera (salvo numpy). Nadie importa `material/`.

## Instalación (una sola vez)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`-e` instala el paquete "editable": los cambios en `alc/` se ven al instante, y
`from alc import ...` funciona desde cualquier carpeta (scripts, notebooks, TP).

## Uso

```python
from alc import calculaQR, normaExacta, metpot2k
```

```bash
pytest                    # todos los tests
pytest tests/test_L05.py  # un labo
pytest -k normaMatMC      # una función
```

## Convenciones

- **Nombres**: exactamente los de la sección "Módulo ALC" de cada PDF, aunque
  mezclen camelCase y snake_case. Los tests de la cátedra dependen de eso.
- **Sin `@`, `np.matmul` ni `np.dot`** dentro de `alc/` (lo prohíbe la cátedra).
  Usar `productoInterno`, `productoExterno`, `calcularAx` y
  `multiplicacionDeMatrices` de `alc/basicos.py`. En los tests sí se puede.
- **Imports internos explícitos**, p. ej. `from .basicos import calcularAx`.
  Así, al copiar una función para entregar, se ve de qué otras depende.
- **Nada de tests sueltos ni `print` a nivel módulo en `alc/`**: los tests van en `tests/`.

## Agregar un labo nuevo (ej. L08)

1. PDF y archivos de la cátedra → `material/L08/` (sin tocarlos).
2. Funciones del Módulo ALC → `alc/<tema>.py`, y sumarlas en `alc/__init__.py`.
3. Tests de la cátedra → `tests/test_L08.py` (envolver los `assert` en funciones `test_...`).
4. Experimentos / gráficos → `labos/L08-<tema>/`.

## Estado del Módulo ALC

| Labo | Archivo | Pendiente |
|------|---------|-----------|
| L00 | `basicos.py` | — |
| L01 | `maquina.py` | — |
| L02 | `transformaciones.py` | — |
| L03 | `normas.py` | — |
| L04 | `lu.py` | — |
| L05 | `qr.py` | — |
| L06 | `autovalores.py` | `diagRH` |
| L07 | `markov.py` | todo (`nucleo` depende de `diagRH`) |
