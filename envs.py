from dataclasses import dataclass 
import numpy

@dataclass
class GridWorld:
    w: int 
    h: int

    @property
    def __get_states(self) -> int:
        return self.w * self.h