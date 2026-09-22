"""Prefer the app venv over Azure's bundled OpenTelemetry.

Azure App Service / Monitor injects `/agents/python/common` onto sys.path.
That tree is an older `opentelemetry.sdk` without `ReadableLogRecord`.
Chroma then imports a 1.39+ OTLP exporter from antenv, which needs that name.

This module must be imported before chromadb or any opentelemetry package.
"""

from __future__ import annotations

import os
import site
import sys


_AZURE_PATH_MARKERS = (
    "/agents/python/",
    "applicationinsights",
    "azuremonitor",
)


def _is_azure_agent_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return any(marker in normalized for marker in _AZURE_PATH_MARKERS)


def _site_package_dirs() -> list[str]:
    dirs: list[str] = []
    try:
        dirs.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            dirs.append(user_site)
    except Exception:
        pass
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    prefix_site = os.path.join(sys.prefix, "lib", version, "site-packages")
    if os.path.isdir(prefix_site):
        dirs.append(prefix_site)
    seen: set[str] = set()
    unique: list[str] = []
    for directory in dirs:
        if directory and directory not in seen:
            seen.add(directory)
            unique.append(directory)
    return unique


def prefer_app_packages() -> None:
    site_dirs = _site_package_dirs()
    azure_paths = [path for path in sys.path if path and _is_azure_agent_path(path)]
    other_paths = [path for path in sys.path if path and not _is_azure_agent_path(path)]

    ordered: list[str] = []
    for path in [*site_dirs, *other_paths, *azure_paths]:
        if path not in ordered:
            ordered.append(path)
    sys.path[:] = ordered

    for name in list(sys.modules):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            del sys.modules[name]


prefer_app_packages()
