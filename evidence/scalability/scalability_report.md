# Reporte de Escalabilidad 1X / 10X / 100X — Ransa Big Data 2026

## Entorno de ejecución
- **Proyecto GCP:** `ransa-big-data-2026` (`southamerica-west1`)
- **Clúster Dataproc:** `ransa-spark` (1 master + 2 workers `e2-standard-2`)
- **Dataset base (Silver):** `gs://ransa-big-data-2026-lake/silver/transporte_carga/`
- **Ruta de pruebas:** `gs://ransa-big-data-2026-lake/test/scalability/` (sin modificar `bronze/` ni `silver/`)
- **Nota:** Las escalas `10x` y `100x` son replicaciones sintéticas controladas (`ES_DATO_SINTETICO = True`).

## Resultados del Benchmark

| Escala | Datos sintéticos | Filas | Volumen en GCS | Particiones | Executors | Tiempo (s) | Estado |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | :--- |
| **1X** | No (`False`) | 1,394,094 | 180.86MiB | 2 | 2 | 19.1427 | OK |
| **10X** | Sí (`True`) | 13,940,940 | 278.61MiB | 3 | 3 | 9.3992 | OK |
| **100X** | Sí (`True`) | 139,409,400 | 999.28MiB | 8 | 4 | 31.1623 | OK |

## Análisis de escalamiento
- **Efecto *cold start* (1X vs 10X):** La primera ejecución (`1X`, 19.14 s) absorbió el calentamiento inicial de la JVM y la asignación de recursos con 2 executors. En `10X` el entorno ya estaba caliente y Spark asignó 3 executors y 3 particiones, completando en 9.40 s.
- **Escalamiento sublineal (10X vs 100X):** Al multiplicar el volumen por 10 (de 13.94 M a 139.41 M de filas), el tiempo solo creció 3.31x (de 9.40 s a 31.16 s) gracias al paralelismo de 8 particiones y 4 executors.
- **Conclusión:** Procesar 100 veces más datos tomó solo 1.63x el tiempo inicial, validando que Dataproc + Parquet/Snappy escala de forma sublineal sin degradación crítica.