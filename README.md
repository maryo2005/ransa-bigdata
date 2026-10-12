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
  - Ejecución y coordinación de trabajos Spark

#### Nodo Worker
- Hostname: `ransa-worker1`
- Rol: nodo trabajador
- Servicios principales:
  - HDFS DataNode
  - YARN NodeManager
  - Ejecución distribuida de tareas Spark

En el nodo trabajador se configuraron los siguientes componentes:

- Debian 12
- OpenJDK 17
- Hadoop 3.3.6
- SSH Server
- Git
- Python 3
- PySpark
- Tailscale para la comunicación remota entre nodos

## Comunicación entre nodos

Los nodos se encuentran conectados mediante Tailscale, permitiendo establecer comunicación entre las máquinas virtuales aunque se encuentren en redes físicas diferentes.

La resolución de nombres utilizada es:

```text
ransa-master
ransa-worker1
