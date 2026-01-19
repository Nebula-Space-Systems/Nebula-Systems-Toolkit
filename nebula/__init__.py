import os
from typing import Optional

import jdk4py
import orekit_jpype

NEBULA_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OREKIT_DATA = os.path.normpath(
    os.path.join(NEBULA_ROOT_DIR, "..", "data", "orekit-data")
)

_orekit_data_configured: Optional[str] = None


def ensure_setup(data_path: Optional[str] = None) -> str:
    """Ensure the JVM is running and Orekit data is registered.

    Args:
        data_path: Optional directory containing the Orekit data tree. Defaults
            to the packaged ``data/orekit-data`` directory.

    Returns:
        The path that was registered with Orekit.

    Raises:
        FileNotFoundError: If the provided data path does not exist.
    """

    global _orekit_data_configured

    path = os.path.normpath(data_path or _DEFAULT_OREKIT_DATA)

    if _orekit_data_configured == path:
        return path

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Orekit data directory not found: {path}")

    # jdk4py exposes a managed JDK; set JAVA_HOME if the user hasn't already.
    os.environ.setdefault("JAVA_HOME", str(jdk4py.JAVA_HOME))

    # initVM is idempotent; safe to call repeatedly.
    orekit_jpype.initVM()

    from orekit_jpype.pyhelpers import setup_orekit_curdir

    setup_orekit_curdir(filename=path)
    _orekit_data_configured = path

    return path


# Keep previous behavior: configure Orekit data on import using the packaged set.
ensure_setup()
