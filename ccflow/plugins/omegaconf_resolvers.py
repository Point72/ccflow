import re
import sys
from datetime import date, datetime, time
from pathlib import Path
from socket import gethostname
from subprocess import check_output
from typing import Dict, List, Optional, Union
from zoneinfo import ZoneInfo

from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import TypeAdapter

# Import this file to register the resolvers with OmegaConf
__all__ = ("register_omegaconf_resolver",)


def register_omegaconf_resolver(name: str, func, replace: bool = True, prefix: Optional[str] = "cc") -> None:
    if OmegaConf.has_resolver(name) and replace:
        OmegaConf.clear_resolver(name)
        if prefix and not name.startswith(f"{prefix}."):
            # Also register with prefix if not already present
            OmegaConf.clear_resolver(f"{prefix}.{name}")
    if not OmegaConf.has_resolver(name) or replace:
        OmegaConf.register_new_resolver(name, func)
        if prefix and not name.startswith(f"{prefix}."):
            OmegaConf.register_new_resolver(f"{prefix}.{name}", func)


def today_resolver(tz_name: Optional[str] = None) -> str:
    if tz_name is None:
        return datetime.now().date()  # we use local time if None
    tz = ZoneInfo(tz_name)
    local_time = datetime.now(ZoneInfo("UTC")).astimezone(tz)
    return local_time.date()


# pattern to match command line parameters in the form of key=value and strip the leading '+' or '-' if present
_param_pattern = re.compile(r"^[+-]*(?P<key>[^=]+)=(?P<value>.+)$")


def date_resolver(
    dt: Union[DictConfig, ListConfig, str, datetime, date],
    datetime_format: str,
    date_format: str = None,
) -> str:
    """Convert Omegaconf config/datetime/string into a `datetime_format` formatted string.
    If the the input `dt` parameter has zero time (e.g. 2023-10-01 00:00:00) and the optional `date_format`
    is provided, then it will return the date formatted as `date_format`.
    Args:
        dt (Union[DictConfig, ListConfig, str, datetime, date]): The datetime object or omegaconf config to format.
        datetime_format (str): The format string for the datetime.
        date_format (str, optional): The format string for the date. Defaults to None.
    Returns:
        str: The formatted date or datetime string.
    """
    if isinstance(dt, (DictConfig, ListConfig)):
        dt = OmegaConf.to_container(dt, resolve=True)
    if isinstance(dt, dict):
        # e.g. DatetimeContext object, {"dt": datetime(...)}
        dt = list(dt.values())[0]
    if isinstance(dt, list):
        dt = dt[0]
    dt = TypeAdapter(Union[datetime, date]).validate_python(dt)
    if dt.time() == time(0, 0) and date_format:
        return dt.strftime(date_format)
    return dt.strftime(datetime_format)


def param_resolver(param_name: str = None, args: List[str] = None) -> Union[str, Dict[str, str]]:
    """Resolve a parameter from the command line arguments in the form of key=value.
    If the parameter is not found, it returns an empty string.
    If `param_name` is not provided, it returns a dictionary of all parameters found.
    Args:
        param_name (str, optional): The name of the parameter to resolve. Defaults to None.
        args (List[str], optional): The list of command line arguments to search for parameters. Defaults to None,
            which uses `sys.argv`.
    Returns:
        Union[str, Dict[str, str]]: The value of the parameter if `param_name` is provided,
        or a dictionary of all parameters found in the command line arguments.
    """
    if args is None:
        args = sys.argv
    params = {match.group("key"): match.group("value") for arg in args for match in [_param_pattern.match(arg)] if match}
    if param_name:
        return params.get(param_name, "")
    return params


# Register a resolver to return the current date in a specified timezone. If none, uses local time
register_omegaconf_resolver("today_at_tz", today_resolver)

# Taking a list of keys to create a dictionary and an element to populate each entry in the dictionary with,
# produce a dictionary from list element to static dict elements
register_omegaconf_resolver("list_to_static_dict", lambda keys, static_val: {x: static_val for x in keys})

# Merge a list of tuples together to build a dictionary, can be used as a workaround for OmegaConf being
# unable to interpolate var values used as dictionary keys
register_omegaconf_resolver(
    "dict_from_tuples",
    lambda tuples: {k: v for k, v in tuples},
)

register_omegaconf_resolver("trim_null_values", lambda dict_val: {k: v for k, v in dict_val.items() if v is not None})

# Perform replacements in strings
register_omegaconf_resolver("replace", lambda input_val, orig_val, replace_val: input_val.replace(orig_val, replace_val))

# Perform split in strings
register_omegaconf_resolver("split", lambda input_val, sep: input_val.split(sep))

# Perform join in strings
register_omegaconf_resolver("join", lambda input_val, sep: sep.join(input_val))

# Provides a path to the current user's home directory
register_omegaconf_resolver("user_home", lambda: str(Path.home()))

# Provides the machine hostname without having to use oc.env:HOSTNAME
register_omegaconf_resolver("hostname", lambda: gethostname())

# Returns a boolean value indicating whether the value provided is None or an empty string
register_omegaconf_resolver("is_none_or_empty", lambda x: x is None or x == "")

# Negates the provided value
register_omegaconf_resolver("is_not", lambda x: not x)

# Allows the toggling of values depending on a boolean flag supplied
register_omegaconf_resolver("if_else", lambda value, value_true, value_false: value_true if value else value_false)

# Register a resolver to boolean if an interpolated value is not provided
register_omegaconf_resolver("is_provided", lambda a, *, _parent_: a in _parent_ and _parent_[a] is not None)
register_omegaconf_resolver("is_missing", lambda a, *, _parent_: a not in _parent_)

# Register a resolver to run a command and return the value
register_omegaconf_resolver("cmd", lambda cmd: check_output([cmd], shell=True).decode("utf-8").rstrip("\n"))

# Register a resolver to format a date/datetime object or config into a formatted string
register_omegaconf_resolver("strftime", date_resolver)

# Handle command line parameters passed in the form of key=value
register_omegaconf_resolver("param", param_resolver)
