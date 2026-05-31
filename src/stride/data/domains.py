from pathlib import Path


def read_domain_list(path: Path) -> list[str]:
    domains: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        domains.append(value)
    if not domains:
        msg = f"domain list is empty: {path}"
        raise ValueError(msg)
    return domains


def mdcath_h5_path(input_root: Path, domain_id: str) -> Path:
    direct = input_root / f"mdcath_dataset_{domain_id}.h5"
    nested = input_root / "data" / f"mdcath_dataset_{domain_id}.h5"
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    return nested
