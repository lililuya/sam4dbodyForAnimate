def resolve_completion_batch_size(configured_batch_size):
    if configured_batch_size is None:
        return 1
    return max(1, int(configured_batch_size))


def resolve_decode_chunk_size(configured_decode_chunk_size, *, num_frames):
    if num_frames <= 0:
        return 1
    if configured_decode_chunk_size is None:
        return int(num_frames)
    return max(1, min(int(configured_decode_chunk_size), int(num_frames)))


def build_completion_slice(
    *,
    first_occ_idx,
    last_occ_idx,
    total_frames,
    pad_before=2,
    pad_after=2,
    max_occ_len=0,
):
    start = max(0, int(first_occ_idx) - int(pad_before))
    end = min(int(total_frames), int(last_occ_idx) + 1 + int(pad_after))

    if max_occ_len is not None and int(max_occ_len) > 0:
        end = min(end, start + int(max_occ_len))

    return start, end
