from captain_harness.adapters.capx.services import CapXHttpToolServices, CapXServiceEndpoints, _parse_molmo_point

FULL_OUTPUTS = [
    '<points coords="1 1 207 177">blue mat</points>',
    '<points coords="1 1 499 555 2 610 120">gripper</points>',
    '<point x="10.5" y="20.3" alt="red block">red block</point>',
    "There are none.",
]


def _stopped(text):
    """What vLLM returns with stop=['">'] and include_stop_str_in_output=True."""
    index = text.find('">')
    return text if index < 0 else text[:index + 2]


def test_stop_after_coords_parses_like_the_full_answer():
    for text in FULL_OUTPUTS:
        assert _parse_molmo_point(_stopped(text)) == _parse_molmo_point(text)


class _Recorder:
    def __init__(self):
        self.payloads = []

    def request(self, method, url, *, payload=None, timeout_s=None):
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": '<points coords="1 1 500 250">'}}]}


def test_stop_fields_only_when_enabled():
    import numpy as np
    image = np.zeros((4, 8, 3), np.uint8)
    for enabled in (False, True):
        transport = _Recorder()
        services = CapXHttpToolServices(CapXServiceEndpoints(molmo="http://molmo/v1"), transport=transport,
                                        molmo_stop_after_coords=enabled)
        assert services.point(image, "the cube") == (4.0, 1.0)
        payload = transport.payloads[0]
        assert ("stop" in payload) is enabled and ("include_stop_str_in_output" in payload) is enabled
