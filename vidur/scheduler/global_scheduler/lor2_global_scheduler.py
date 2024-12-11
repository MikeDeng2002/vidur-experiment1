from typing import List, Tuple
from math import ceil

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class LOR2GlobalScheduler(BaseGlobalScheduler):
    """
    Least outstanding requests (LOR) global scheduler.
    """

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        # 创建内存使用率映射
        pending_requests_map = {
            replica_scheduler.replica_id: 
            replica_scheduler.num_pending_requests+1 / replica_scheduler._flops
            for replica_scheduler in self._replica_schedulers.values()
        }

        # using a very simple implementation here, to keep wiring simple
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = min(pending_requests_map.items(), key=lambda x: x[1])[0]
            pending_requests_map[replica_id] += 1 / self._replica_schedulers[replica_id]._flops  
            request_mapping.append((replica_id, request))

        return request_mapping