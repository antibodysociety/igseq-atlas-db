_HEAVY_PREFIXES = frozenset({"H", "VH", "CH1", "CH2", "CH3"})
_LIGHT_PREFIXES = frozenset({"L", "VL", "V"})


def infer_chain_type(chain_description: str, css_suffix: str) -> str:
    """Infer semantic chain type from IMGT chain description.

    For canonical heavy/light chains the first dash-delimited token is
    sufficient.  SCFV and FUSION are returned as their own types, except
    FUSION uses the CSS suffix (h/l) as a discriminator because the
    description alone does not distinguish the two.
    """
    prefix = chain_description.split("-")[0].strip().upper()
    if prefix in _HEAVY_PREFIXES:
        return "h"
    if prefix in _LIGHT_PREFIXES:
        return "l"
    if "SCFV" in prefix:
        return "scfv"
    if "FUSION" in prefix:
        return css_suffix if css_suffix in {"h", "l"} else "fusion"
    return "noncanonical"
