
from typing import Tuple

def get_gymInfo(env_name: str) -> Tuple[Tuple, int, int, float]:
    '''
    gymの環境の情報を返す

    state_size, action_size, action_kinds, clearScoreThreshold
    '''

    state_size: Tuple = tuple([0])
    action_kinds: int = 0
    action_size: int = 0
    clearScoreThreshold: float = 0.0

    if env_name == "ALE/Breakout-v5":
        state_size = (210, 160, 3)
        action_kinds = 4
        action_size = 1
        clearScoreThreshold = 100.0
    
    return state_size, action_size, action_kinds, clearScoreThreshold