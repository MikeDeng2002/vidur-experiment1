from vidur.types.base_int_enum import BaseIntEnum


class GlobalSchedulerType(BaseIntEnum):
    RANDOM = 1
    ROUND_ROBIN = 2
    LOR = 3
    LOR1 = 4
    LOR2 = 5
    LOR3 = 6
