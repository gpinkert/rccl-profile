from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional

VALID_COLLECTIVES = {
    "all", "all_gather", "all_reduce", "alltoall", "alltoallv", "scatter",
    "broadcast", "gather", "reduce", "reduce_scatter", "sendrecv"
}

VALID_OPS = {"sum", "prod", "min", "max", "avg", "mulsum", "all"}
VALID_DATATYPES = {"int8", "uint8", "int32", "uint32", "int64",
                   "uint64", "half", "float", "double", "bloat16",
                   "fp8_e4m3", "fp8_e5m2"}

VALID_STEP_TYPES = {"multiple", "increment"}

@dataclass
class StepDetail:
    type: str
    value: int

    def __post_init__(self):
        if self.type not in VALID_STEP_TYPES:
            raise ValueError(f"Invalid step type: {self.type}. Must be one of {VALID_STEP_TYPES}.")

@dataclass
class EnvVar:
    name: str
    value: str

@dataclass
class Configuration:
    collectives: List[str]
    start_size: Union[int, str]
    end_size: Union[int, str]
    iterations: int
    operation: List[str]
    step_detail: Optional[StepDetail]
    datatypes: List[str]
    ENV_VARS: List[EnvVar] = field(default_factory=list)
    gpus_per_thread: Optional[int] = 1

    def __post_init__(self):
        for collective in self.collectives:
            if collective not in VALID_COLLECTIVES and collective != "all":
                raise ValueError(f"Invalid collective: {collective}. Must be one of {VALID_COLLECTIVES}.")

        if isinstance(self.start_size, int) and self.start_size < 0:
            raise ValueError("start_size cannot be less than 0")
        if isinstance(self.end_size, int) and self.end_size < 0:
            raise ValueError("end_size cannot be less than 0")
        for datatype in self.datatypes:
            if datatype not in VALID_DATATYPES:
                raise ValueError(f"DataType {datatype} not supported")

    @staticmethod
    def from_dict(d: Dict) -> 'Configuration':
        step_detail_dict = d.get("step_details", [{}])[0]
        step_detail = StepDetail(**step_detail_dict)
        env_vars = []
        for var in d.get("ENV_VARS", []):
            if isinstance(var, dict):
                print(var)
                env_vars.append(EnvVar(name=var['id'], value=var["value"]))
            else:
                raise ValueError(f"Each ENV_VAR entry must be a dict, got: {var}")
        return Configuration(
            collectives=d["collectives"],
            start_size=d["start_size"],
            end_size=d["end_size"],
            iterations=d["iterations"],
            operation=d["operation"],
            step_detail=step_detail,
            datatypes=d["datatypes"],
            ENV_VARS=env_vars,
            gpus_per_thread=d.get("gpus_per_thread", 1),
        )
