from typing import List, Tuple
from math import ceil

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class LOR3GlobalScheduler(BaseGlobalScheduler):
    """
    Least outstanding requests (LOR) global scheduler.
    """

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        # 创建内存使用率映射
        pending_requests_map = {
            replica_scheduler.replica_id: replica_scheduler.num_allocated_blocks
            for replica_scheduler in self._replica_schedulers.values()
        }

        # using a very simple implementation here, to keep wiring simple
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = min(pending_requests_map.items(), key=lambda x: x[1])[0]
            pending_requests_map[replica_id] += request._num_prefill_tokens // 16
            request_mapping.append((replica_id, request))

        return request_mapping