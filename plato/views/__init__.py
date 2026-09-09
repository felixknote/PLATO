"""The two analysis arms.

``browser`` is the plate-map-aware thumbnail grid; ``explorer`` is the
embedding-space scatter. Both are plain QWidgets driven by the same
``Session``, so the shell can host them as tabs without either knowing the
other exists.
"""
