# Prueba de procesamiento PySpark en Google Cloud Dataproc

## Configuración del clúster

- Nombre del clúster: `ransa-spark`
- Región: `southamerica-west1`
- Nodo master: 1
- Workers: 2
- Imagen de Dataproc: `2.2.89-debian12`

## Objetivo de la prueba

Verificar que el clúster de Google Cloud Dataproc pueda ejecutar correctamente un trabajo PySpark y escribir los resultados en Google Cloud Storage.

## Prueba ejecutada

Se generó un DataFrame con:

- 1,000,000 registros
- 2 particiones de salida

## Ruta de salida

`gs://ransa-big-data-2026-lake/temp/smoke_test`

## Resultado

`SUCCESS`

La prueba confirmó que Apache Spark ejecutado mediante Google Cloud Dataproc puede procesar datos y escribir resultados correctamente en el bucket del proyecto.

Esta prueba funciona como validación inicial de la infraestructura antes de ejecutar los procesos reales de limpieza, integración y transformación de los datasets del proyecto Ransa Comercial.