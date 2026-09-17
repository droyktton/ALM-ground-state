# ALM ground state — Anharmonic Larkin Model

Solver exacto del estado fundamental del **Anharmonic Larkin Model (ALM)**,
más un conjunto de herramientas para caracterizar su rugosidad estadística
(exponente ζ) mediante el factor de estructura promediado sobre desorden.

## El modelo

Interfaz elástica 1D `u_i` sometida a una fuerza de desorden aleatoria `f_i`
(media nula), con energía

```
H[u] = sum_i [ (c/2) s_i^2 + (1/2n) |s_i|^(2n) - f_i u_i ],   s_i = u_{i+1} - u_i
```

donde `c >= 0` es la constante elástica armónica y `n > 1` controla el término
anarmónico (`n=2` da un término cuártico `|s|^4/4`). La condición de
equilibrio de fuerzas es

```
sigma_i - sigma_{i-1} = f_i,      sigma(s) = c*s + |s|^(2n-2) s
```

Este modelo se resuelve **exactamente**: integrando la relación anterior, el
problema de minimizar `H` en las `L` variables `u_i` se reduce a encontrar la
raíz de una única función escalar `G(C)` (una constante de integración `C`).
Una vez hallada esa raíz por bisección (`scipy.optimize.brentq`), las
pendientes `s_i` se obtienen invirtiendo la relación constitutiva
`sigma = s(sigma)` (analíticamente si `c=0`, con Newton salvaguardado si
`c>0`), y la interfaz se reconstruye por suma acumulada.

`alm.py` implementa esta construcción para dos condiciones de contorno:

- **Periódicas** (`solve_ground_state`): requiere resolver la raíz escalar `C`
  (hay un grado de libertad global). Se usa cuando se analiza la interfaz vía
  FFT.
- **Libres / abiertas** (`solve_ground_state_free`): las condiciones de borde
  de tensión nula en los extremos fijan `C = 0` exactamente, sin necesidad de
  búsqueda de raíz. Se usa junto con una DCT (en vez de una FFT) porque la
  interfaz no es periódica.

En ambos casos se exige desorden de media nula (`sum_i f_i = 0`), condición
necesaria para que exista una configuración de equilibrio estático.

## Archivos

| Archivo | Qué hace |
|---|---|
| `alm.py` | Núcleo del proyecto: genera el desorden (`sample_disorder`, con formas gaussiana/uniforme/bimodal, todas normalizadas a la misma varianza `Delta`) y resuelve el estado fundamental exacto (`solve_ground_state`, `solve_ground_state_free`). Los demás scripts importan estas funciones. Ejecutado directamente (`python3 alm.py`), corre una demo: genera una realización, resuelve el estado fundamental, verifica el balance de fuerzas y grafica `u_i`, `s_i` y `F_i`. |
| `run_structure_factor.py` | Para un `(L, n, c, Delta)` dado, genera muchas realizaciones de desorden, resuelve el estado fundamental de cada una, y promedia el factor de estructura `S(q) = <|û(q)|^2>/L`. Ajusta por regresión log-log el exponente de rugosidad espectral `ζ_s` de `S(q) ~ q^{-(1+2ζ_s)}`, y valida el resultado calculando el ancho cuadrático `W²` de dos formas independientes (directo en espacio real, y vía el teorema de Parseval a partir de `S(q)`) — deben coincidir. |
| `plot_illustrative_configs.py` | Genera una figura ilustrativa (no promediada) del perfil de altura y de pendiente normalizados, `[h(x)-h̄]/σ_h` y `m/σ_m`, para distintos valores de `p = 1/(2n-1)` (usa el caso puro `c=0`, donde `p` es el exponente de la relación constitutiva). |
| `plot_zeta_vs_n.py` | Lee un CSV producido por un barrido en `(L, n)` (típicamente `scan_results.csv`, generado por `scan_sweep_bc.sh`), y grafica `ζ_s(n, L)` (del factor de estructura) junto con `ζ(n)` obtenido ajustando `W²(L) ~ L^{2ζ}` a través de los distintos tamaños `L`, comparando ambos contra curvas de teoría. |
| `scan_sweep_bc.sh` | Script bash que barre varios valores de `n`, `L` y ambas condiciones de contorno (periódica y libre), llama a `run_structure_factor.py` para cada combinación, y junta los resultados en `scan_results.csv` (una fila por corrida, con `ζ_s`, `W²`, etc.). Es el generador del CSV que consume `plot_zeta_vs_n.py`. |

## Flujo de trabajo típico

```
alm.py  ──(genera configuraciones de estado fundamental)──▶
    ├─▶ run_structure_factor.py   (análisis para un único (L,n): S(q), ζ_s, W²)
    ├─▶ plot_illustrative_configs.py  (perfiles ilustrativos de u(x), s(x))
    └─▶ scan_sweep_bc.sh  ──▶ scan_results.csv ──▶ plot_zeta_vs_n.py  (ζ(n) vs teoría)
```

1. **Demo rápida / sanity check** del solver:
   ```bash
   python3 alm.py
   ```
   Genera una realización con `L=2000`, `c=1`, `n=2`, resuelve el estado
   fundamental, imprime chequeos de consistencia (cierre periódico, balance
   de fuerzas) y guarda `alm_ground_state.png`.

2. **Factor de estructura y exponente ζ_s** para un tamaño y anarmonicidad
   dados:
   ```bash
   python3 run_structure_factor.py -L 8192 -n 2.0 --samples 300 --bc periodic
   python3 run_structure_factor.py -L 8192 -n 3.0 -c 1.0 --delta 1.0 --bc free
   ```
   Opciones relevantes: `-L` tamaño del sistema, `-n` exponente anarmónico,
   `-c` constante elástica, `--delta` varianza del desorden, `--samples`
   número de realizaciones a promediar, `--dist {gaussian,uniform,bimodal}`,
   `--bc {periodic,free}`, ventana de ajuste vía `--qfrac`/`--qmin`/`--qmax`/
   `--kmin`/`--kmax`, `-o` archivo de salida del gráfico, `--csv` para volcar
   `(q, S(q))`, `--no-plot`/`--no-show` para correr sin abrir ventana gráfica.
   Al final imprime una línea `RESULT ...` con todos los valores clave, pensada
   para ser parseada por scripts (es lo que hace `scan_sweep_bc.sh`).

3. **Configuraciones ilustrativas** para distintos `p = 1/(2n-1)`:
   ```bash
   python3 plot_illustrative_configs.py --p 0.2 4.0 -L 16384 --no-show
   ```

4. **Barrido completo y comparación con teoría**:
   ```bash
   ./scan_sweep_bc.sh                       # genera scan_results.csv (puede tardar bastante)
   python3 plot_zeta_vs_n.py --csv scan_results.csv --Lmin 8192 --no-show
   ```
   `plot_zeta_vs_n.py` acepta `--xaxis {p,n}`, `--Lmin`/`--Lmax` para excluir
   tamaños chicos del ajuste de tamaño finito, `--zeta-s-L` para mostrar
   `ζ_s` solo a un tamaño fijo, y `--summary-csv` para guardar la tabla de
   `ζ(n)` ajustada.

## Requisitos

```bash
pip install numpy scipy matplotlib pandas
```

Python 3.8+ recomendado.
