"""Let torchvision accept the agent view's masked dataset files (FML-bench Claude Code sandbox).

In the agent sandbox the hidden test samples of the task's data are blanked, which changes the
files' md5. torchvision verifies CIFAR/SVHN files against hard-coded md5s and would refuse
them (or try to re-download the originals). $FMLBENCH_MASKED_MD5 points to a JSON table
{original md5: masked md5}; once torchvision.datasets.utils is imported, its check_md5 also
accepts the masked md5 for a file whose original md5 is expected. Nothing else changes.
"""
import importlib.abc
import importlib.util
import json
import os
import sys

_TABLE = os.environ.get("FMLBENCH_MASKED_MD5")
_TARGET = "torchvision.datasets.utils"


def _patch(mod, aliases):
    orig = mod.check_md5

    def check_md5(fpath, md5, **kwargs):
        if orig(fpath, md5, **kwargs):
            return True
        alt = aliases.get(md5)
        return alt is not None and orig(fpath, alt, **kwargs)

    mod.check_md5 = check_md5


class _Hook(importlib.abc.MetaPathFinder):
    def __init__(self, aliases):
        self.aliases = aliases

    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            pass
        if spec is None or spec.loader is None:
            return spec
        loader, aliases = spec.loader, self.aliases
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module, aliases)

        loader.exec_module = exec_and_patch
        return spec


if _TABLE and os.path.isfile(_TABLE):
    try:
        with open(_TABLE) as f:
            _aliases = json.load(f)
    except (OSError, ValueError):
        _aliases = {}
    if _aliases:
        sys.meta_path.insert(0, _Hook(_aliases))
