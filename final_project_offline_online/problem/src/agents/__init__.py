from .fql_agent import FQLAgent
from .sacbc_agent import SACBCAgent
from .qsm_agent import QSMAgent
from .dsrl_agent import DSRLAgent
from .ifql_agent import IFQLAgent
from .world_model_agent import WorldModelAgent

agents = {
    "fql": FQLAgent,
    "sacbc": SACBCAgent,
    "qsm": QSMAgent,
    "dsrl": DSRLAgent,
    "ifql": IFQLAgent,
    "world_model": WorldModelAgent,
}
