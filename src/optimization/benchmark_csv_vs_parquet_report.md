# Benchmark controlado: CSV vs. Parquet+Snappy (`transporte_carga`)

## 1. Entorno y dataset
- **Proyecto GCP:** `ransa-big-data-2026`
- **Región:** `southamerica-west1`
- **Bucket:** `gs://ransa-big-data-2026-lake/`
- **Clúster de referencia:** `ransa-spark` (1 master + 2 workers `e2-standard-2`, autoeliminación por inactividad)
- **Dataset evaluado:** `transporte_carga` (1,394,094 registros; ~180.30 MiB en Silver Parquet+Snappy)
- **Metodología:** Misma operación de lectura y procesamiento sobre el mismo dataset, con repeticiones múltiples reportando la **mediana** de tiempo (no el mejor tiempo aislado).

## 2. Resultados medidos
| Formato | Mediana de tiempo | Mejora relativa frente a CSV | Registros | Tamaño Silver (referencia) |
| --- | --- | --- | --- | --- |
| CSV | `9.1150 s` | Base (1.00x) | 1,394,094 | — |
| Parquet + Snappy | `2.9045 s` | ~3.14x más rápido | 1,394,094 | ~180.30 MiB |

## 3. Conclusión técnica
- En la prueba controlada sobre `transporte_carga` (1,394,094 registros), **Parquet + Snappy** redujo la mediana de ejecución de **9.1150 s** a **2.9045 s** (~3.14x más rápido que CSV).
- La ventaja de rendimiento se sustenta en los tiempos medidos bajo la misma operación y no únicamente en la reducción de tamaño físico o particiones en disco.
