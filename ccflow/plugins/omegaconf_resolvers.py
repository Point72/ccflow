from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from omegaconf import OmegaConf

# Import this file to register the resolvers with OmegaConf
__all__ = ()


def today_resolver(tz_name: Optional[str] = None) -> str:
    if tz_name is None:
        return datetime.now().date()  # we use local time if None
    tz = ZoneInfo(tz_name)
    local_time = datetime.now(ZoneInfo("UTC")).astimezone(tz)
    return local_time.date()


# Register a resolver to return the current date in a specified timezone. If none, uses local time
OmegaConf.register_new_resolver("today_at_tz", today_resolver)

# Taking a list of keys to create a dictionary and an element to populate each entry in the dictionary with,
# produce a dictionary from list element to static dict elements
OmegaConf.register_new_resolver("list_to_static_dict", lambda keys, static_val: {x: static_val for x in keys})

# Merge a list of tuples together to build a dictionary, can be used as a workaround for OmegaConf being
# unable to interpolate var values used as dictionary keys
OmegaConf.register_new_resolver(
    "dict_from_tuples",
    lambda tuples: {k: v for k, v in tuples},
)

OmegaConf.register_new_resolver("trim_null_values", lambda dict_val: {k: v for k, v in dict_val.items() if v is not None})

# Perform replacements in strings
OmegaConf.register_new_resolver("replace", lambda input_val, orig_val, replace_val: input_val.replace(orig_val, replace_val))

# Provides a path to the current user's home directory
OmegaConf.register_new_resolver("user_home", lambda: str(Path.home()))

# Returns a boolean value indicating whether the value provided is None or an empty string
OmegaConf.register_new_resolver("is_none_or_empty", lambda x: x is None or x == "")

# Negates the provided value
OmegaConf.register_new_resolver("is_not", lambda x: not x)

# Allows the toggling of values depending on a boolean flag supplied
OmegaConf.register_new_resolver("if_else", lambda value, value_true, value_false: value_true if value else value_false)

# Register a resolver to boolean if an interpolated value is not provided
OmegaConf.register_new_resolver("is_missing", lambda a, *, _parent_: a not in _parent_)
