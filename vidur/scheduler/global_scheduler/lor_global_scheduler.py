from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class LORGlobalScheduler(BaseGlobalScheduler):
    """
    Least outstanding requests (LOR) global scheduler.
    """

    
    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        # 创建包含pending requests和FLOPS信息的映射
        pending_requests_map = {
            replica_scheduler.replica_id: {
                'pending': replica_scheduler.num_pending_requests,
                'flops': replica_scheduler._flops
            }
            for replica_scheduler in self._replica_schedulers.values()
        }

        while self._request_queue:
            request = self._request_queue.pop(0)
            # 首先按照pending requests排序，如果相同则按FLOPS降序排序
            replica_id = min(
                pending_requests_map.items(),
                key=lambda x: (x[1]['pending'], -x[1]['flops'])  # 负号使FLOPS降序排序
            )[0]
            
            pending_requests_map[replica_id]['pending'] += 1
            request_mapping.append((replica_id, request))

        return request_mapping
