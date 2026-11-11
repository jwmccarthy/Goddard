"""Coverage-first reset selection, without modifying any recorded state."""
import hashlib
import heapq
from collections import Counter, deque

import numpy as np

POSITION_SCALE = np.array((4108., 6000., 2076.))


def reset_bins(scenes):
    ego = scenes[:, 9:30]
    position = ego[:, :3] * POSITION_SCALE
    spatial = np.floor(position / (512, 640, 256)).astype(np.int16)
    ball_relative = (scenes[:, :3] - ego[:, :3]) * POSITION_SCALE
    opponent_relative = (scenes[:, 30:33] - ego[:, :3]) * POSITION_SCALE
    heading = np.arctan2(ego[:, 10], ego[:, 9])
    context = np.column_stack((
        np.floor(position / 96),
        np.floor(ball_relative / (192, 192, 128)),
        np.floor(opponent_relative / (384, 384, 256)),
        np.floor(ego[:, 3:6] * 2300 / 300),
        np.floor(heading / (np.pi / 8)),
        np.floor(ego[:, 14] * 4),
        np.floor(ego[:, 15] * 5),
        np.floor(ego[:, 6:9] * 5.5),
    )).astype(np.int16)
    return spatial, context


def different_scene(left, right):
    """Within one event, require a material positional or kinematic difference."""
    for start, velocity in ((0, 6000), (9, 2300), (30, 2300)):
        if np.linalg.norm((left[start:start + 3] - right[start:start + 3]) * POSITION_SCALE) >= 100:
            return True
        if np.linalg.norm((left[start + 3:start + 6] - right[start + 3:start + 6]) * velocity) >= 250:
            return True
        if start:
            for axis in (9, 12):
                if np.dot(left[start + axis:start + axis + 3], right[start + axis:start + axis + 3]) < np.cos(np.pi / 12):
                    return True
            if abs(left[start + 15] - right[start + 15]) >= .1:
                return True
    return False


def diverse_reset_choices(items, limit, rng, spacing=2, event_limit=8):
    """Round-robin spatial cells, then events; reject local near duplicates.

    Resets store single states, so they do not need discriminator-history-sized
    spacing. At 30 Hz, two dense frames separate candidates by at least 67 ms;
    the state-difference gate adds a stronger within-event novelty requirement.
    """
    cells = {}
    for item in items:
        event = (item["player"], item["period"], item["event"])
        cells.setdefault(item["spatial"], {}).setdefault(event, []).append(item)
    keys = list(cells)
    rng.shuffle(keys)
    queues = {}
    for cell, events in cells.items():
        order = list(events)
        rng.shuffle(order)
        queues[cell] = deque(order)
        for values in events.values():
            rng.shuffle(values)
    chosen, event_states, contexts, occupied = [], {}, {}, {}
    while keys and len(chosen) < limit:
        remaining = []
        for cell in keys:
            queue = queues[cell]
            accepted = False
            while queue and not accepted:
                event = queue.popleft()
                pool = cells[cell][event]
                prior = event_states.setdefault(event, [])
                if len(prior) >= event_limit:
                    continue
                seen_contexts = contexts.setdefault(event, set())
                while pool:
                    item = pool.pop()
                    identity = (item["player"], item["period"])
                    used = occupied.setdefault(identity, set())
                    if any(t in used for t in range(item["end"] - spacing + 1, item["end"] + spacing)):
                        continue
                    if item["context"] in seen_contexts:
                        continue
                    scene = item["rows"][item["end"], :51]
                    if not all(different_scene(scene, other) for other in prior):
                        continue
                    chosen.append(item)
                    prior.append(scene)
                    used.add(item["end"])
                    seen_contexts.add(item["context"])
                    accepted = True
                    break
                if pool and len(prior) < event_limit:
                    queue.append(event)
            if queue:
                remaining.append(cell)
            if len(chosen) == limit:
                break
        keys = remaining
    return chosen


def reset_signature(scene, internal):
    state = np.concatenate((scene, internal.ravel())).astype('<f4')
    state[state == 0] = 0  # Signed zero is not a different reset state.
    return hashlib.blake2b(state.tobytes(), digest_size=16).hexdigest().encode('ascii')


def balanced_spatial_select(pools, count, seed, players, pairs, used, player_cap, pair_cap,
                            match_cap, signatures):
    """Match round-robin with global least-used-cell priority and cell caps."""
    rng = np.random.default_rng(seed)
    keys = sorted(pools)
    rng.shuffle(keys)
    cells, heaps, spatial_counts = {}, {}, Counter()
    all_cells = {item["spatial"] for pool in pools.values() for item in pool}
    spatial_cap = max(1, int(np.ceil(count * .02)), int(np.ceil(3 * count / max(1, len(all_cells)))))
    for key, pool in pools.items():
        grouped = {}
        for item in pool:
            grouped.setdefault(item["spatial"], []).append(item)
        heap = []
        for cell, values in grouped.items():
            rng.shuffle(values)
            heapq.heappush(heap, (0, float(rng.random()), cell))
        cells[key], heaps[key] = grouped, heap
    selected, matches = [], Counter()
    while keys and len(selected) < count:
        remaining = []
        for key in keys:
            heap, accepted = heaps[key], False
            while heap and matches[key] < match_cap and not accepted:
                previous, tie, cell = heapq.heappop(heap)
                current = spatial_counts[cell]
                if current >= spatial_cap:
                    continue
                if previous != current:
                    heapq.heappush(heap, (current, tie, cell))
                    continue
                pool = cells[key][cell]
                while pool:
                    item = pool.pop()
                    identity = (key, item["player"], item["period"], item["end"])
                    if (identity in used or item["signature"] in signatures
                            or players[item["player"]] >= player_cap or pairs[item["pair"]] >= pair_cap):
                        continue
                    used.add(identity)
                    signatures.add(item["signature"])
                    selected.append(item)
                    players[item["player"]] += 1
                    pairs[item["pair"]] += 1
                    spatial_counts[cell] += 1
                    matches[key] += 1
                    accepted = True
                    break
                if pool and spatial_counts[cell] < spatial_cap:
                    heapq.heappush(heap, (spatial_counts[cell], float(rng.random()), cell))
            if heap and matches[key] < match_cap:
                remaining.append(key)
            if len(selected) == count:
                break
        keys = remaining
    if len(selected) != count:
        raise ValueError(f"stratum underfilled: requested {count}, found {len(selected)} under spatial/diversity caps")
    return selected
