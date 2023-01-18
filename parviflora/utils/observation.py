from typing import Union

import numpy as np
from numpy.typing import NDArray


def unsqueeze_observation(
    observation: Union[dict[str, NDArray], NDArray]
) -> Union[dict[str, NDArray], NDArray]:
    if isinstance(observation, dict):
        return {k: v[np.newaxis, :] for k, v in observation.items()}
    else:
        return observation[np.newaxis, :]
