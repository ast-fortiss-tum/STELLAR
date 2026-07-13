def convert_pymoo_time_to_seconds(pymoo_time: str) -> int:
    res = pymoo_time.split(":")
    H = int(res[0])
    M = int(res[1])
    s = int(res[2])
    return 3600*H + 60*M + s
    