# Modelado predictivo con Gradient-Boosted Trees (`GBTClassifier`) — Kiara

## 1. Metodología, particiones y reglas de control
* **Selección de hiperparámetros:** Entrenamiento sobre `FIT` (`4,501` filas) de `gs://ransa-big-data-2026-lake/gold/dataset_tuning_fit_validation_2026/` usando `CLASS_WEIGHT_TUNING`; selección por **PR-AUC** (`areaUnderPR`) sobre `VALIDATION` (`1,126` filas), sin consultar `TEST`.
* **Entrenamiento final y evaluación:** Reentrenamiento con la mejor configuración (`maxIter=50`, `maxDepth=6`, `stepSize=0.05`, `subsamplingRate=0.8`, `seed=42`) sobre `TRAIN` (`5,627` filas) de `gs://ransa-big-data-2026-lake/gold/dataset_modelado_final_2026/` con `CLASS_WEIGHT`, y evaluación única sobre `TEST` (`1,431` filas: `145` positivos reales y `1,286` negativos reales).
* **Prevención de fuga y trazabilidad:** `ID_TRAMO` y `RUTA_JOIN` excluidos del vector de `features` y conservados únicamente para trazabilidad. Artefactos persistidos en `gs://ransa-big-data-2026-lake/models/kiara/` (`gbt_model_final`, `test_predictions`, `metrics_report`) y reporte respaldado en `results/kiara/gbt_metrics_report.json`.

## 2. Resultados en `VALIDATION` y `TEST`

| Escenario | PR-AUC | ROC-AUC | Precision | Recall | F1-Score | Accuracy | Matriz (`TP/FP/FN/TN`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VAL (`50, 6, 0.05` — Ganadora)** | 0.2050 | 0.6706 | 0.2418 | 0.3190 | 0.2751 | 0.8268 | — |
| **TEST — GBTClassifier (`kiara`)** | 0.1730 | 0.6342 | 0.1853 | 0.3310 | 0.2376 | 0.7848 | `48 / 211 / 97 / 1075` |
| **TEST — Baseline (`Siempre 0`)** | 0.1730 | 0.6342 | 0.0000 | 0.0000 | 0.0000 | 0.8987 | `0 / 0 / 145 / 1286` |
