"""Study-design helpers for Olink experiments.

Port of ``OlinkAnalyze::olink_plate_randomizer``.

Randomizes a sample manifest across plates so that batch / plate
effects are not confounded with the study design. Two modes:

* **Sample randomization** — every sample placed at a random well
  (used when there is no repeated-subject structure).
* **Subject-keeping randomization** — all samples belonging to one
  subject are kept on the same plate (pass ``subject_col``); plates are
  filled subject-by-subject and wells scrambled within each plate.

Control wells (``num_ctrl`` per plate) are reserved as
``CONTROL_SAMPLE`` and excluded from the usable spots.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


_ROWS = list("ABCDEFGH")


def _well_layout(plate_size: int):
    """Return the (row, column) wells of a plate in column-major order."""
    n_cols = plate_size // 8
    return [(r, c) for c in range(1, n_cols + 1) for r in _ROWS], n_cols


def olink_plate_randomizer(
    manifest: pd.DataFrame,
    plate_size: int = 96,
    subject_col: Optional[str] = None,
    num_ctrl: int = 8,
    iterations: int = 500,
    seed: Optional[int] = None,
    sample_col: str = "SampleID",
) -> pd.DataFrame:
    """Randomize samples across plates.

    Parameters
    ----------
    manifest :
        DataFrame with at least a ``SampleID`` column.
    plate_size :
        48 or 96 wells.
    subject_col :
        If given, keep all of a subject's samples on the same plate.
    num_ctrl :
        Number of control wells reserved per plate.
    iterations :
        Max re-tries of the subject-keeping packing before giving up.
    seed :
        RNG seed for reproducibility.
    sample_col :
        Sample-ID column name.

    Returns
    -------
    pandas.DataFrame
        ``manifest`` plus ``plate``, ``column``, ``row`` and ``well``
        columns. Control wells are appended as ``CONTROL_SAMPLE`` rows.
    """
    if plate_size not in (48, 96):
        raise ValueError("plate_size must be 48 or 96.")
    if num_ctrl < 1 or int(num_ctrl) != num_ctrl:
        raise ValueError("num_ctrl must be a positive integer.")
    if sample_col not in manifest.columns:
        raise ValueError(f"manifest must contain a '{sample_col}' column.")
    if manifest[sample_col].isna().any():
        raise ValueError("No NA allowed in the SampleID column.")

    rng = np.random.default_rng(seed)
    wells, n_cols = _well_layout(plate_size)
    spots_per_plate = plate_size - num_ctrl
    man = manifest.reset_index(drop=True).copy()

    def _well_str(row, col):
        return f"{row}{col}"

    if subject_col is None:
        n = len(man)
        n_plates = int(np.ceil(n / spots_per_plate))
        order = rng.permutation(n)
        assign = []
        for plate in range(1, n_plates + 1):
            chunk = order[(plate - 1) * spots_per_plate: plate * spots_per_plate]
            usable = wells[:len(chunk)]
            for idx, (row, col) in zip(chunk, usable):
                assign.append((idx, plate, col, row))
        amap = {idx: (p, c, r) for idx, p, c, r in assign}
        man["plate"] = [f"Plate {amap[i][0]}" for i in range(n)]
        man["column"] = [amap[i][1] for i in range(n)]
        man["row"] = [amap[i][2] for i in range(n)]
    else:
        if subject_col not in man.columns:
            raise ValueError(f"subject_col '{subject_col}' not in manifest.")
        if man[subject_col].isna().any():
            raise ValueError("No NA allowed in the subject column.")
        subjects = man[subject_col].astype(str)
        groups = {s: man.index[subjects == s].tolist()
                  for s in subjects.unique()}
        n_plates_guess = int(np.ceil(len(man) / spots_per_plate))

        placed = None
        for _ in range(iterations):
            n_plates = n_plates_guess
            order = list(rng.permutation(list(groups.keys())))
            # plate -> remaining capacity
            cap = {p: spots_per_plate for p in range(1, n_plates + 1)}
            assign = {}
            ok = True
            for sub in order:
                members = groups[sub]
                target = None
                for p in range(1, n_plates + 1):
                    if cap[p] >= len(members):
                        target = p
                        break
                if target is None:
                    n_plates += 1
                    cap[n_plates] = spots_per_plate
                    target = n_plates
                    if cap[target] < len(members):
                        ok = False
                        break
                for m in members:
                    assign[m] = target
                cap[target] -= len(members)
            if ok:
                placed = assign
                break
        if placed is None:
            raise RuntimeError(
                "Could not keep all subjects on the same plate; "
                "increase iterations or num_ctrl."
            )
        # Scramble wells within each plate.
        plate_members: dict = {}
        for idx, p in placed.items():
            plate_members.setdefault(p, []).append(idx)
        man["plate"] = ""
        man["column"] = 0
        man["row"] = ""
        for p, members in plate_members.items():
            members = list(members)
            chosen = rng.permutation(len(members))
            usable = wells[:len(members)]
            for slot, m in zip(chosen, members):
                row, col = usable[slot]
                man.loc[m, "plate"] = f"Plate {p}"
                man.loc[m, "column"] = col
                man.loc[m, "row"] = row

    man["well"] = [_well_str(r, c) for r, c in zip(man["row"], man["column"])]

    # Append control wells: take the unused wells of each plate.
    ctrl_rows = []
    for plate, grp in man.groupby("plate"):
        used = set(zip(grp["row"], grp["column"]))
        free = [w for w in wells if w not in used]
        for row, col in free[:num_ctrl]:
            ctrl_rows.append({
                sample_col: "CONTROL_SAMPLE", "plate": plate,
                "column": col, "row": row, "well": _well_str(row, col),
            })
    if ctrl_rows:
        man = pd.concat([man, pd.DataFrame(ctrl_rows)],
                        axis=0, ignore_index=True)
    return man.sort_values(["plate", "column", "row"]).reset_index(drop=True)
