import json

from vidur.config import BaseRequestGeneratorConfig, ClusterConfig, MetricsConfig
from vidur.entities.base_entity import BaseEntity
from vidur.entities.replica import Replica
from vidur.logger import init_logger

logger = init_logger(__name__)


class Cluster(BaseEntity):
    def __init__(
        self,
        cluster_config: ClusterConfig,
        metrics_config: MetricsConfig,
        generator_config: BaseRequestGeneratorConfig,
    ) -> None:
        self._id = Cluster.generate_id()
        self._config = cluster_config
        
       
        
        self._output_dir = metrics_config.output_dir
        self._replicas = {}
        self._replica_index_map = {}
        device_counts = {}

        for i, config in enumerate(cluster_config.replica_configs):
           
            
            device_counts[config.device] = device_counts.get(config.device, 0) + 1
            replica = Replica(config, generator_config)
            self._replicas[replica.id] = replica
            self._replica_index_map[i] = replica.id

   
        for device, count in device_counts.items():
            logger.info(f"  {device}: {count} replicas")

        if metrics_config.write_json_trace:
            self._write_cluster_info_to_file()

    @property
    def replicas(self):
        return self._replicas

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "num_replicas": len(self._replicas),
        }

    def _write_cluster_info_to_file(self) -> None:
        replica_dicts = [replica.to_dict() for replica in self._replicas.values()]
        cluster_info = {"replicas": replica_dicts}

        cluster_file = f"{self._output_dir}/cluster.json"
        with open(cluster_file, "w") as f:
            json.dump(cluster_info, f)

    def get_replica_by_index(self, index: int) -> Replica:
        """通过配置序号获取副本"""
        replica_id = self._replica_index_map.get(index)
        if replica_id is None:
            raise ValueError(f"No replica found for index {index}")
        return self._replicas[replica_id]

    def get_replica_device(self, index: int) -> str:
        """通过配置序号获取副本设备类型"""
        replica = self.get_replica_by_index(index)
        return replica.device
