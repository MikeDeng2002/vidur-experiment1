from vidur.scheduler.global_scheduler.lor_global_scheduler import LORGlobalScheduler
from vidur.scheduler.global_scheduler.random_global_scheduler import (
    RandomGlobalScheduler,
)
from vidur.scheduler.global_scheduler.lor1_golbal_scheduler import LOR1GlobalScheduler
from vidur.scheduler.global_scheduler.round_robin_global_scheduler import (
    RoundRobinGlobalScheduler,
)
from vidur.scheduler.global_scheduler.lor2_global_scheduler import LOR2GlobalScheduler
from vidur.scheduler.global_scheduler.lor3_global_scheduler import LOR3GlobalScheduler
from vidur.types import GlobalSchedulerType
from vidur.utils.base_registry import BaseRegistry


class GlobalSchedulerRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> GlobalSchedulerType:
        return GlobalSchedulerType.from_str(key_str)


GlobalSchedulerRegistry.register(GlobalSchedulerType.RANDOM, RandomGlobalScheduler)
GlobalSchedulerRegistry.register(
    GlobalSchedulerType.ROUND_ROBIN, RoundRobinGlobalScheduler
)
GlobalSchedulerRegistry.register(GlobalSchedulerType.LOR, LORGlobalScheduler)
GlobalSchedulerRegistry.register(GlobalSchedulerType.LOR1, LOR1GlobalScheduler)
GlobalSchedulerRegistry.register(GlobalSchedulerType.LOR2, LOR2GlobalScheduler)
GlobalSchedulerRegistry.register(GlobalSchedulerType.LOR3, LOR3GlobalScheduler)
