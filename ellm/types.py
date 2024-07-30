from dataclasses import dataclass


@dataclass
class PreferenceData:
    prompt: str
    chosen_response: str
    rejected_response: str
