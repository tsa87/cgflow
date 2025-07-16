import datetime
import time
from pathlib import Path

# fmt: off
TARGETS = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1",
    "GBA", "IDH1", "KAT2A", "MAPK1", "MTORC1",
    "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
]
# fmt: on
current_timestamp = time.time()

pocket_logs = {}
for target in TARGETS:
    dir = Path("./logs/exp1-redocking/redock-catalog") / target
    for seed in range(3):
        pose_dir = dir / f"seed-{seed}" / "pose"
        if not pose_dir.exists():
            continue
        N = 0
        T = 0
        T0 = 0
        i = 0
        for i in range(1, 1001):
            oracle_path = pose_dir / f"oracle{i}.sdf"
            if oracle_path.exists():
                T = oracle_path.stat().st_ctime
                N += 1
                if i == 1:
                    T0 = T
            else:
                break
        if N == 1000:
            continue
        from_1_to_i = T - T0
        from_i_to_1000 = (T - T0) * (1000 - i) / (i - 1)

        timestamp = datetime.datetime.fromtimestamp(T).strftime("%H:%M:%S")
        time_diff = datetime.timedelta(seconds=int(current_timestamp - T))
        total_time = datetime.timedelta(seconds=int(from_1_to_i))
        remain_time = datetime.timedelta(seconds=int(from_i_to_1000))
        print(f"{target:<16}{seed}\t{N}\t{timestamp}\t{time_diff} \t{total_time} \t{remain_time}")
