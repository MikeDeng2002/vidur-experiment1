from typing import List, Tuple
from math import ceil

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class LOR1GlobalScheduler(BaseGlobalScheduler):
    """
    Least outstanding requests (LOR) global scheduler.
    """

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        # 创建内存使用率映射
        memory_usage_map = {
            replica_scheduler.replica_id: replica_scheduler.memory_usage_percent
            for replica_scheduler in self._replica_schedulers.values()
        }

        # 基于内存使用百分比进行调度
        while self._request_queue:
            request = self._request_queue.pop(0)
            # 选择内存使用率最低的副本
            replica_id = min(memory_usage_map.items(), key=lambda x: x[1])[0]
            # 更新内存使用率：(预填充token数/16)占总块数的比例
            memory_usage_map[replica_id] += (
                ceil(request._num_prefill_tokens // 16) * 100 //
                self._replica_schedulers[replica_id]._config.num_blocks
            )
            request_mapping.append((replica_id, request))

        return request_mapping
