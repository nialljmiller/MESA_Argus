"""python -m mesa_argus.sources [NAME]: list the source generators and their keys."""

import sys

from . import describe, get, registry

if __name__ == "__main__":
    gens = [get(n) for n in sys.argv[1:]] or sorted(registry().values(), key=lambda g: (g.timing, g.name))
    print("\n\n".join(describe(g) for g in gens))
