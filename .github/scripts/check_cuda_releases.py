"""Warn about newer official CUDA redistributions without downloading binaries.

>>> newer_releases([(12, 9, 1), (13, 4, 2)], [(12, 9, 2), (13, 4, 2)])
[(12, 9, 2)]
>>> newer_releases([(12, 9, 2)], [(12, 9, 2), (13, 4, 2), (13, 5, 0)])
[(13, 4, 2), (13, 5, 0)]
>>> newer_releases([(13, 4, 2)], [(13, 4, 1), (13, 4, 2)])
[]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote
from urllib.request import urlopen

REDIST = "https://developer.download.nvidia.com/compute/cuda/redist/"


def fetch(url):
    with urlopen(url, timeout=30) as response:
        return response.read().decode()


def version(value):
    return tuple(map(int, value.split(".")))


def parse_releases(html):
    """Read stable manifests from NVIDIA's single- or double-quoted links.

    >>> parse_releases("<a href='redistrib_13.4.2.json'>new</a>")
    [(13, 4, 2)]
    >>> parse_releases('<a href="redistrib_12.9.2.json">old</a><a href="redistrib_14.0.0rc1.json">preview</a>')
    [(12, 9, 2)]
    """
    return sorted({version(v) for v in re.findall(r"""href=['"]redistrib_(\d+\.\d+\.\d+)\.json['"]""", html)})


def newer_releases(configured, available):
    newest = {release[:2]: release for release in sorted(available)}
    updates = []
    for minor, release in newest.items():
        pins = [pin for pin in configured if pin[0] == minor[0]]
        if pins:
            changed = minor > max(pin[:2] for pin in pins) or any(pin[:2] == minor and pin < release for pin in pins)
        else:
            changed = minor[0] > max(pin[0] for pin in configured)
        if changed:
            updates.append(release)
    return sorted(updates)


def torch_wheel_channels(release, index, fetch_index, minimum=0):
    """Only report channels with wheels for the exact stable PyTorch release.

    >>> pages = {132: 'torch-2.14.1%2Bcu132-cp314-cp314t-linux_x86_64.whl',
    ...          134: 'torch-2.15.0%2Bcu134-cp314-cp314-linux_x86_64.whl',
    ...          136: 'torch-2.16.0rc1%2Bcu136-cp314-cp314-linux_x86_64.whl'}
    >>> torch_wheel_channels('2.15.0', '<a href="cu132/">old</a><a href="cu134/">new</a><a href="cu136/">preview</a>', pages.__getitem__)
    [134]
    >>> torch_wheel_channels('2.16.0', '<a href="cu136/">preview</a>', pages.__getitem__)
    []
    """
    channels = sorted({int(v) for v in re.findall(r"cu(\d+)/", index)})
    return [channel for channel in channels if channel > minimum and f"torch-{release}+cu{channel}-" in unquote(fetch_index(channel))]


def warn(message):
    message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::warning::{message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workflow", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        text = "\n".join(path.read_text() for path in args.workflow)
        configured = [version(v) for v in re.findall(
            r"(?:CUDA-|cuda-version:\s*(?:\n\s*default:\s*)?(?:\[\s*)?[\"']?)(\d+\.\d+\.\d+)", text
        )]
        channels = [int(v) for v in re.findall(r"cu(\d+)", text)]
        # Reusable build jobs select the channel from the SDK major/minor.
        if not channels:
            channels = [132 if pin[0] == 13 else pin[0] * 10 + pin[1] for pin in configured]
        if not configured or not channels:
            raise ValueError("No CUDA toolkit pins or Torch wheel channels found")
        available = parse_releases(fetch(REDIST))
        if not available:
            raise ValueError("NVIDIA's index contained no stable CUDA releases")
        print(f"Latest stable CUDA toolkit: {'.'.join(map(str, available[-1]))}")
        for release in newer_releases(configured, available):
            warn(f"{args.workflow}: CUDA {'.'.join(map(str, release))} is newer than the SDK matrix; review GPU compatibility.")
        torch = json.loads(fetch("https://pypi.org/pypi/torch/json"))["info"]["version"]
        print(f"Latest stable PyTorch: {torch}")
        torch_pins = [version(v) for v in re.findall(
            r"(?:torch-version:\s*(?:\n\s*default:\s*)?(?:\[\s*)?[\"']?)(\d+\.\d+\.\d+)", text
        )]
        if torch_pins and version(torch) > max(torch_pins):
            warn(f"Stable Torch {torch} is newer than the wheel matrix; review Python and CUDA wheel availability.")
        updated = torch_wheel_channels(
            torch,
            fetch("https://download.pytorch.org/whl/"),
            lambda channel: fetch(f"https://download.pytorch.org/whl/cu{channel}/torch/"),
            max(channels),
        )
        for channel in updated:
            warn(f"{args.workflow}: stable Torch {torch} wheels now support cu{channel}; review the wheel matrix.")
        if not updated:
            print(f"No newer stable Torch CUDA wheel channel beyond cu{max(channels)}.")
    except (OSError, ValueError) as error:
        warn(f"Release availability could not be verified: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
