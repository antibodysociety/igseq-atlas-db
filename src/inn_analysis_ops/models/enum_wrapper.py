import sys

# if python version is greater than 3.11, use StrEnum else use Enum
if sys.version_info.major == 3 and sys.version_info.minor >= 11:
    from enum import StrEnum as EnumBase
else:
    from enum import Enum as EnumBase

if sys.version_info.major == 3 and sys.version_info.minor >= 11:

    class StrEnumBase(EnumBase):
        pass

else:

    class StrEnumBase(str, EnumBase):
        pass
