"""
Actividad 16. Análisis de desempeño de algoritmo paralelo
Simulación meteorológica 2D con descomposición por franjas y Ghost Cells.

200711 - Carlos Andrés Hernández Díaz
200871 - Carlos Jesus Graniel Gonzalez
195721 - Rogelio Sotomayor Carrasco

Requisitos:
    pip install numpy

El programa ejecuta las configuraciones de 1, 2, 4 y 8 procesos, mide el
tiempo total de pared y calcula aceleración y eficiencia respecto a 1 proceso.
"""

import argparse
import multiprocessing as mp
import time
import statistics
import numpy as np


def repartir_filas(N, n_proc):
    """Distribuye todas las filas, incluyendo el residuo, entre procesos."""
    base, residuo = divmod(N, n_proc)
    return [base + (1 if i < residuo else 0) for i in range(n_proc)]


def inicializacion_subdominio(N, inicio_global, filas_locales, perturbacion=1.01):
    """
    Crea una matriz local de (filas_locales + 2) x (N + 2).
    La primera y última fila, y la primera y última columna, son halos
    (Ghost Cells). El dominio físico ocupa [1:-1, 1:-1].
    """
    grid = np.zeros((filas_locales + 2, N + 2), dtype=np.float64)
    filas_globales = (np.arange(inicio_global, inicio_global + filas_locales)
                      * perturbacion)
    columnas_globales = np.arange(N)

    rr, cc = np.meshgrid(filas_globales, columnas_globales, indexing="ij")
    distancia = np.sqrt((rr - 50.0) ** 2 + (cc - 50.0) ** 2)
    grid[1:-1, 1:-1] = 100.0 * np.exp(-distancia / 120.0)
    return grid


def intercambio_ghost_cells(grid, pipe_top, pipe_bottom):
    """Intercambia filas limítrofes con los procesos vecinos."""
    if pipe_top is not None:
        pipe_top.send(grid[1, 1:-1].copy())
        grid[0, 1:-1] = pipe_top.recv()

    if pipe_bottom is not None:
        pipe_bottom.send(grid[-2, 1:-1].copy())
        grid[-1, 1:-1] = pipe_bottom.recv()


def calcular_paso_atmosferico(grid):
    """Aplica un stencil de cuatro vecinos sobre las celdas interiores."""
    siguiente = grid.copy()
    siguiente[1:-1, 1:-1] = 0.25 * (
        grid[0:-2, 1:-1] +       # Norte
        grid[2:, 1:-1] +         # Sur
        grid[1:-1, 0:-2] +       # Oeste
        grid[1:-1, 2:]           # Este
    )
    return siguiente


def worker_spmd(rank, N, steps, inicio, filas_locales,
                pipe_top, pipe_bottom, result_pipe, perturbacion):
    """Cada proceso ejecuta el mismo ciclo sobre su subdominio."""
    grid = inicializacion_subdominio(N, inicio, filas_locales, perturbacion)

    for _ in range(steps):
        intercambio_ghost_cells(grid, pipe_top, pipe_bottom)
        grid = calcular_paso_atmosferico(grid)

    # Se devuelve una suma de comprobación para confirmar que hubo cálculo.
    checksum = float(np.sum(grid[1:-1, 1:-1]))
    result_pipe.send((rank, checksum))
    result_pipe.close()
    if pipe_top is not None:
        pipe_top.close()
    if pipe_bottom is not None:
        pipe_bottom.close()


def ejecutar_simulacion(n_proc, N, steps, perturbacion=1.01):
    """Crea procesos, conecta vecinos y mide el tiempo total de ejecución."""
    filas_por_proceso = repartir_filas(N, n_proc)
    procesos = []
    resultados = [None] * n_proc

    # Un Pipe dúplex conecta cada pareja de procesos vecinos.
    conexiones = [mp.Pipe(duplex=True) for _ in range(n_proc - 1)]
    pipes_top = [None] * n_proc
    pipes_bottom = [None] * n_proc

    for i, (extremo_a, extremo_b) in enumerate(conexiones):
        pipes_bottom[i] = extremo_a
        pipes_top[i + 1] = extremo_b

    # Pipes independientes para devolver el resultado al proceso principal.
    result_pipes = [mp.Pipe(duplex=False) for _ in range(n_proc)]

    inicio_global = 0
    t0 = time.perf_counter()

    for rank in range(n_proc):
        filas_locales = filas_por_proceso[rank]
        receptor, emisor = result_pipes[rank]
        p = mp.Process(
            target=worker_spmd,
            args=(rank, N, steps, inicio_global, filas_locales,
                  pipes_top[rank], pipes_bottom[rank], emisor, perturbacion)
        )
        procesos.append(p)
        p.start()
        emisor.close()
        inicio_global += filas_locales

    for rank, (receptor, _) in enumerate(result_pipes):
        resultados[rank] = receptor.recv()
        receptor.close()

    for p in procesos:
        p.join()

    tiempo_total = time.perf_counter() - t0

    errores = [p.exitcode for p in procesos if p.exitcode != 0]
    if errores:
        raise RuntimeError(f"Uno o más procesos terminaron con error: {errores}")

    # Cerrar en el proceso principal los extremos de comunicación.
    for a, b in conexiones:
        a.close()
        b.close()

    return tiempo_total, resultados


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de simulación meteorológica paralela."
    )
    parser.add_argument("--N", type=int, default=500,
                        help="Resolución del dominio N x N (predeterminado: 500).")
    parser.add_argument("--steps", type=int, default=30,
                        help="Número de iteraciones (predeterminado: 30).")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repeticiones por configuración (predeterminado: 3).")
    parser.add_argument("--perturbacion", type=float, default=1.01)
    args = parser.parse_args()

    if args.N < 8 or args.steps < 1 or args.repeats < 1:
        parser.error("N debe ser >= 8, steps y repeats deben ser >= 1.")

    mp.freeze_support()
    procesos_a_probar = [p for p in (1, 2, 4, 8) if p <= args.N]
    tiempos = {}

    print(f"Simulación: N={args.N}, iteraciones={args.steps}, "
          f"repeticiones={args.repeats}")
    print("Los tiempos incluyen creación, comunicación y cierre de procesos.")

    for p in procesos_a_probar:
        mediciones = []
        for repeticion in range(args.repeats):
            tiempo, _ = ejecutar_simulacion(
                p, args.N, args.steps, args.perturbacion
            )
            mediciones.append(tiempo)
            print(f"P={p}, repetición {repeticion + 1}: {tiempo:.6f} s")
        tiempos[p] = statistics.median(mediciones)

    t1 = tiempos[1]
    print("\\nRESULTADOS (mediana de las repeticiones)")
    print(f"{'Procesos':>10} {'TP (s)':>12} {'Aceleración':>14} {'Eficiencia':>13}")
    for p in procesos_a_probar:
        aceleracion = t1 / tiempos[p]
        eficiencia = aceleracion / p
        print(f"{p:>10} {tiempos[p]:>12.6f} "
              f"{aceleracion:>14.3f} {eficiencia * 100:>12.2f}%")

    print("\\nNota: la aceleración y eficiencia se calculan respecto al tiempo "
          "medido con 1 proceso en esta misma computadora.")


if __name__ == "__main__":
    main()
