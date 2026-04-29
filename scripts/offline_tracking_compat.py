from collections.abc import Sequence


def unpack_propagate_output(output):
    if not isinstance(output, Sequence):
        raise ValueError("propagate_in_video output must be a sequence with at least 4 values")
    if len(output) < 4:
        raise ValueError("propagate_in_video output must contain at least 4 values")
    return output[0], output[1], output[2], output[3]
