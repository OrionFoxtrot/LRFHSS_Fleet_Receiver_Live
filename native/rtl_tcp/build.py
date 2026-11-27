"""Build rtl_tcp_fleet.exe with 64-bit MinGW-w64 (POSIX threads) and Git on PATH."""
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = HERE / "rtl-sdr-2.0.2.tar.gz"
SOURCE_SHA256 = "d69943eb32df742bc38a00ce6615e41250fd57851174e5ff916ec31e9e9e68e9"


def main():
    if sha256(ARCHIVE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unexpected rtl-sdr source archive")
    with tempfile.TemporaryDirectory(dir=ROOT) as staging:
        with tarfile.open(ARCHIVE) as archive:
            archive.extractall(staging, filter="data")
        source = Path(staging) / "rtl-sdr-2.0.2"
        subprocess.run(["git", "apply", "--no-index", str(HERE / "transfer-size.patch")],
                       cwd=source, check=True, env={**os.environ, "GIT_CEILING_DIRECTORIES": staging})
        exe = Path(staging) / "rtl_tcp_fleet.exe"
        subprocess.run(["gcc", "-O2", "-s", "-Iinclude", "src/rtl_tcp.c",
                        "src/convenience/convenience.c", "src/getopt/getopt.c",
                        str(ROOT / "librtlsdr.dll"), "-lwinpthread", "-lws2_32",
                        "-Wl,--no-insert-timestamp", "-o", str(exe)], cwd=source, check=True)
        exe.replace(ROOT / exe.name)


if __name__ == "__main__":
    main()
