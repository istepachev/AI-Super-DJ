"""SuperDJ: original planning and orchestration, external audio backends."""
__version__ = "0.23.4"

from .resources import configure_math_threads as _configure_math_threads
_configure_math_threads()
del _configure_math_threads
