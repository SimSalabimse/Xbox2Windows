"""Check failure diagnostics for hardware adapter selectors in a GPU test executable."""
import os
import subprocess
import sys


def check(args, env, allow_missing_factory=False):
    missing = "x2w-no-such-hardware-adapter"
    for selector, message in (("x" * 128, "XBOX2WINDOWS_ADAPTER is longer than 127 characters"),
                              (missing, f"No D3D12 hardware adapter matches XBOX2WINDOWS_ADAPTER={missing}")):
        run = subprocess.run(args, env=dict(env, XBOX2WINDOWS_ADAPTER=selector), capture_output=True, text=True)
        if allow_missing_factory and selector == missing and run.returncode == 77:
            return 77
        if run.returncode != 4 or message not in run.stderr:
            raise RuntimeError(f"expected {message!r}, got exit {run.returncode}\n{run.stdout}{run.stderr}")
    print("overlong and unmatched hardware adapter filters rejected")
    return 0


if __name__ == "__main__":
    sys.exit(check(sys.argv[1:], os.environ))
