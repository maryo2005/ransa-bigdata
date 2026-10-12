# Big Data y Analítica para la optimización de rutas logísticas

## Empresa de aplicación

Ransa Comercial S.A.

## Objetivo

Integrar y analizar fuentes oficiales relacionadas con infraestructura vial, emergencias, accidentes y flujo vehicular para identificar condiciones de riesgo en la Red Vial Nacional y apoyar la planificación logística.

## Arquitectura

Ruta de despliegue: On-premise.

Tecnologías previstas:

- Debian 12
- Hadoop / HDFS
- YARN
- Apache Spark
- PySpark
- JupyterLab
- Apache Parquet
- Git / GitHub

## Integrantes

- Kiara Aguilar Chacón
- Álvaro Arroyo Terán
- Maryory Valencia Galarreta
- Lorena Vásquez Carrillo

---

## Estado actual del proyecto

Actualmente se ha implementado y verificado la infraestructura base del proyecto en un entorno on-premise compuesto por dos máquinas virtuales con Debian 12:

- `ransa-master`
- `ransa-worker1`

Ambos nodos se encuentran conectados mediante una red virtual privada, permitiendo la comunicación remota entre las máquinas.

### Configuración actual del clúster

#### Nodo Master

- Hostname: `ransa-master`
- Rol: nodo principal
- Servicios principales:
  - HDFS NameNode
  - YARN ResourceManager
  - Coordinación de trabajos Spark

#### Nodo Worker

- Hostname: `ransa-worker1`
- Rol: nodo trabajador
- Servicios principales:
  - HDFS DataNode
  - YARN NodeManager
  - Ejecución distribuida de tareas Spark

En el nodo worker se configuraron los siguientes componentes:

- Debian 12
- OpenJDK 17
- Hadoop 3.3.6
- SSH Server
- Git
- Python 3
- Tailscale

---

## Comunicación entre nodos

Los nodos `ransa-master` y `ransa-worker1` se encuentran conectados mediante Tailscale.

Esto permite la comunicación entre ambas máquinas virtuales aunque se encuentren en redes físicas diferentes.

Para verificar la comunicación desde el nodo master hacia el worker:

```bash
ping -c 4 ransa-worker1
