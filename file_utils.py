"""Local output handling, checksums, and file inventory."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
INVENTORY = ROOT / "file_inventory.json"


def zone_id(path):
    try:
        raw = Path(str(path) + ":Zone.Identifier").read_bytes()
    except FileNotFoundError:
        return None
    if len(raw) > 65536:
        raise ValueError(f"Oversized security metadata: {path}")
    text = raw.decode("utf-16") if raw.startswith(b"\xff\xfe") else raw.decode("utf-8")
    active = False
    for line in text.split("\0", 1)[0].splitlines():
        line = line.strip(" \t\r")
        if line.startswith(";"):
            continue
        if line.startswith("["):
            if active:
                return None
            active = bool(re.match(r"^\[[ \t]*ZoneTransfer[ \t]*\]", line, re.I))
        elif active and re.match(r"^ZoneId[ \t]*=", line, re.I):
            value = line.split("=", 1)[1].strip(" \t")
            if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            value = value.lstrip("".join(map(chr, range(33))))
            sign = -1 if value.startswith("-") else 1
            value = value.lstrip("+-") if value.startswith(("+", "-")) else value
            radix = 10
            if value.startswith(("0x", "0o", "0b")):
                radix = {"x": 16, "o": 8, "b": 2}[value[1]]
                value = value[2:]
            digits = ""
            for char in value.lower():
                if char not in "0123456789abcdef"[:radix]:
                    break
                digits += char
            if digits:
                result = (sign * int(digits, radix)) & 0xFFFFFFFF
                return result if result <= 4 else None
            return None
    return None


def mark(path, previous=None):
    path = Path(path)
    zone = zone_id(path)
    if previous == 4 or zone == 4:
        desired = 4
    else:
        desired = 3
    if zone != desired:
        Path(str(path) + ":Zone.Identifier").write_text(
            f"[ZoneTransfer]\r\nZoneId={desired}\r\nHostUrl=about:internet\r\n",
            encoding="utf-8",
        )
    if zone_id(path) != desired:
        raise OSError(f"File metadata verification failed: {path}")


def write_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = zone_id(path) if path.exists() else None
    path.write_bytes(content)
    mark(path, previous)


def write_text(path, text):
    write_bytes(path, text.encode("utf-8"))


def write_json(path, obj):
    write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def snapshot(root=ROOT):
    return {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


def finalize():
    records = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == INVENTORY or ".git" in path.relative_to(ROOT).parts:
            continue
        mark(path)
        records.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    records.append({"path": str(INVENTORY), "sha256": None,
                    "note": "Inventory excludes its own checksum."})
    write_json(INVENTORY, {"updated_utc": datetime.now(timezone.utc).isoformat(), "files": records})
    for record in records:
        if zone_id(record["path"]) not in (3, 4):
            raise OSError(f"File metadata verification failed: {record['path']}")


@contextmanager
def output_session():
    before = snapshot()
    print(f"Before run: {len(before)} files inventoried in {ROOT}", flush=True)
    try:
        yield
    finally:
        after = snapshot()
        for name, checksum in after.items():
            if before.get(name) != checksum:
                mark(name)
        finalize()
        print(f"After run: {len(snapshot())} files inventoried.", flush=True)


if __name__ == "__main__":
    finalize()
    print("File inventory refreshed.")
