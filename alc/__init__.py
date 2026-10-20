"""
Modulo ALC - Algebra Lineal Computacional, 2C 2026.

Se usa importando directamente desde el paquete:

    from alc import calculaQR, normaExacta

Cada tema vive en su propio archivo; este __init__ reexporta todo y sirve
como indice de lo que tiene el modulo. Al agregar una funcion nueva, sumarla
tambien aca.
"""
# L00 - Manipulacion de matrices + productos propios
from .basicos import (
    productoInterno, productoExterno, multiplicacionDeMatrices,
    esCuadrada, triangSup, triangInf, diagonal, traza, traspuesta,
    esSimetrica, calcularAx, intercambiarFilas, sumar_fila_multiplo,
    esDiagonalmenteDominante, matrizCirculante, matrizVandermonde,
    fibonacci, numeroAureo, matrizFiboncacci, matrizHilbert, polinomios,
    row_echelon,
)
# L01 - Numeros de maquina
from .maquina import error, error_relativo, matricesIguales
# L02 - Transformaciones lineales
from .transformaciones import rota, escala, rota_y_escala, afin, trans_afin
# L03 - Normas y condicion
from .normas import norma, normaliza, normaMatMC, normaExacta, condMC, condExacto
# L04 - LU
from .lu import calculaLU, res_tri, inversa, calculaLDV, esSDP
# L05 - QR
from .qr import QR_con_GS, QR_con_HH, calculaQR
# L06 - Autovalores
from .autovalores import metpot2k, diagRH
# L07 - Markov y matrices ralas
from .markov import (
    transiciones_al_azar_continuas, transiciones_al_azar_uniformes,
    nucleo, crea_rala, multiplica_rala_vector,
)
