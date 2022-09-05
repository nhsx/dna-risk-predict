#!/usr/bin/env python3

_MAJOR_VERSION = '0'
_MINOR_VERSION = '1'
_PATCH_VERSION = '0'

_VERSION_SUFFIX = None

# Example, '0.1.0-dev'
__version__ = '.'.join([
    _MAJOR_VERSION,
    _MINOR_VERSION,
    _PATCH_VERSION,
])

if _VERSION_SUFFIX:
    __version__ = f'{__version__}-{_VERSION_SUFFIX}'
