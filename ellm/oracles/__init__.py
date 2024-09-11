from ellm.oracles.pair import PairRMOracle
from ellm.oracles.scalar import ScalarRMOracle


def get_cls(model_name: str):
    if "pairrm" in model_name.lower():
        return PairRMOracle
    if "pythia" in model_name.lower():
        return ScalarRMOracle
    raise NotImplementedError
