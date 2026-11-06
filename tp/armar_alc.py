"""Arma el archivo entregable alc.py juntando en uno solo los modulos de alc/
(laboratorio) y las funciones del TP (calibracion.py y esquinas.py).

Uso (desde la raiz del repo):  python tp/armar_alc.py
Escribe tp/template-alumnos-20261027/alc.py. Hay que volver a correrlo cada vez
que cambie algun modulo de origen. No es parte de la entrega.
"""
import os
import re

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "tp", "template-alumnos-20261027", "alc.py")

# (archivo, titulo de seccion), en orden de dependencias
FUENTES = [
    ("alc/basicos.py", "L00 - Manipulacion de matrices + productos propios"),
    ("alc/maquina.py", "L01 - Numeros de maquina"),
    ("alc/transformaciones.py", "L02 - Transformaciones lineales"),
    ("alc/normas.py", "L03 - Normas y condicion"),
    ("alc/lu.py", "L04 - LU"),
    ("alc/qr.py", "L05 - QR"),
    ("alc/autovalores.py", "L06 - Autovalores"),
    ("alc/markov.py", "L07 - Markov y matrices ralas"),
    ("alc/svd.py", "L08 - SVD"),
    ("tp/calibracion.py", "TP - Calibracion de la camara (Ej. 2 a 5)"),
    ("tp/esquinas.py", "TP - Deteccion de esquinas del tablero (Ej. 3)"),
]


def limpiar(texto, archivo):
    """Saca docstring de modulo, imports internos y duplicados de numpy."""
    # imports entre modulos: 'from .x import ...' (con o sin parentesis) y 'from alc import ...'
    texto = re.sub(r"^from \.\w+ import \([^)]*\)[ \t]*\n", "", texto, flags=re.M)
    texto = re.sub(r"^from (\.\w+|alc) import [^\n(]*\n", "", texto, flags=re.M)
    texto = re.sub(r"^from numpy\.ma\.extras import column_stack\n", "", texto, flags=re.M)
    texto = re.sub(r"^import numpy as np[ \t]*\n", "", texto, flags=re.M)
    # docstring de modulo (al principio)
    texto = re.sub(r'\A\s*(""".*?"""|\'\'\'.*?\'\'\')[ \t]*\n', "", texto, count=1, flags=re.S)
    if archivo == "alc/lu.py":
        # lu.py trae una traspuesta solo para cuadradas; basicos.py trae la general
        texto = re.sub(r"def traspuesta\(A\):.*?return B[ \t]*\n", "", texto, count=1, flags=re.S)
    return texto.strip("\n") + "\n"


def main():
    partes = ['"""\nalc.py - Modulo ALC (Algebra Lineal Computacional, 2C 2026) + funciones del TP.\n\n'
              'Archivo generado con tp/armar_alc.py a partir de los modulos del laboratorio (alc/)\n'
              'y de tp/calibracion.py y tp/esquinas.py. No editar a mano: editar los originales\n'
              'y volver a generarlo.\n"""\nimport numpy as np\n']
    for archivo, titulo in FUENTES:
        with open(os.path.join(RAIZ, archivo), encoding="utf-8") as f:
            texto = limpiar(f.read(), archivo)
        partes.append("\n\n# " + "=" * 70 + "\n# " + titulo + "   (" + archivo + ")\n# " + "=" * 70 + "\n\n" + texto)
    with open(SALIDA, "w", encoding="utf-8") as f:
        f.write("".join(partes))
    print("escrito", SALIDA)


if __name__ == "__main__":
    main()
