import sys
from datetime import date
from pathlib import Path
from socket import gethostname
from unittest.mock import patch

from omegaconf import OmegaConf

from ccflow.plugins.omegaconf_resolvers import register_omegaconf_resolver


def test_today_resolver():
    conf = OmegaConf.create({"date": "${today_at_tz:America/New_York}"})
    assert isinstance(conf["date"], date)


def test_list_to_static_dict():
    conf = OmegaConf.create({"out": "${list_to_static_dict:[a,b,c], 1}"})
    assert conf["out"] == {"a": 1, "b": 1, "c": 1}


def test_dict_from_tuples():
    conf = OmegaConf.create({"out": "${dict_from_tuples:[[a,1],[b,2],[c,3]]}"})
    assert conf["out"] == {"a": 1, "b": 2, "c": 3}


def test_trim_null_values():
    conf = OmegaConf.create({"out": "${trim_null_values:{a:1,b:null,c:3}}"})
    assert conf["out"] == {"a": 1, "c": 3}


def test_replace():
    conf = OmegaConf.create({"out": "${replace:abcde,b,z}"})
    assert conf["out"] == "azcde"


def test_user_home():
    conf = OmegaConf.create({"out": "${user_home:}"})
    assert conf["out"] == str(Path.home())


def test_hostname():
    conf = OmegaConf.create({"out": "${hostname:}"})
    assert conf["out"] == gethostname()


def test_is_none_or_empty():
    conf = OmegaConf.create({"out": "${is_none_or_empty:abc}"})
    assert conf["out"] is False
    conf = OmegaConf.create({"out": "${is_none_or_empty:''}"})
    assert conf["out"] is True
    conf = OmegaConf.create({"out": "${is_none_or_empty:null}"})
    assert conf["out"] is True


def test_is_not():
    conf = OmegaConf.create({"out": "${is_not:true}"})
    assert conf["out"] is False
    conf = OmegaConf.create({"out": "${is_not:false}"})
    assert conf["out"] is True


def test_if_else():
    conf = OmegaConf.create({"out": "${if_else:true,1,2}"})
    assert conf["out"] == 1
    conf = OmegaConf.create({"out": "${if_else:false,1,2}"})
    assert conf["out"] == 2


def test_is_provided():
    conf = OmegaConf.create({"out": "${is_provided:abc}"})
    assert conf["out"] is False
    conf = OmegaConf.create({"parent": {"out": "${is_provided:abc}"}})
    assert conf["parent"]["out"] is False
    conf = OmegaConf.create({"parent": {"abc": True, "out": "${is_provided:abc}"}})
    assert conf["parent"]["out"] is True


def test_is_missing():
    conf = OmegaConf.create({"out": "${is_missing:abc}"})
    assert conf["out"] is True
    conf = OmegaConf.create({"parent": {"out": "${is_missing:abc}"}})
    assert conf["parent"]["out"] is True
    conf = OmegaConf.create({"parent": {"abc": True, "out": "${is_missing:abc}"}})
    assert conf["parent"]["out"] is False


def test_cmd_resolver():
    conf = OmegaConf.create({"out": "${cmd:echo foo}"})
    assert conf["out"] == "foo"


def test_date():
    expected = "2025-06-01"

    conf = OmegaConf.create({"out": "${strftime:2025-06-01, %F_%H-%M-%S}"})
    assert conf["out"] == f"{expected}_00-00-00"

    conf2 = OmegaConf.create({"out": "${strftime:2025-06-01, %F_%H-%M-%S, %F}"})
    assert conf2["out"] == expected


def test_datetime():
    expected = "2025-06-01_12-30-00"

    conf = OmegaConf.create({"out": "${strftime:2025-06-01T12:30, %F_%H-%M-%S}"})
    assert conf["out"] == expected

    conf2 = OmegaConf.create({"out": "${strftime:2025-06-01T12:30, %F_%H-%M-%S, %F}"})
    assert conf2["out"] == expected


def test_str_date():
    expected = "2025-06-01"

    conf = OmegaConf.create({"out": "${strftime:2025-06-01, %F_%H-%M-%S}"})
    assert conf["out"] == f"{expected}_00-00-00"

    conf2 = OmegaConf.create({"out": "${strftime:2025-06-01, %F_%H-%M-%S, %F}"})
    assert conf2["out"] == expected


def test_str_datetime():
    expected = "2025-06-01_12-30-00"

    conf = OmegaConf.create({"out": "${strftime:2025-06-01T12:30:00, %F_%H-%M-%S}"})
    assert conf["out"] == expected

    conf2 = OmegaConf.create({"out": "${strftime:2025-06-01T12:30:00, %F_%H-%M-%S, %F}"})
    assert conf2["out"] == expected


def test_listconfig_date():
    expected = "2025-06-01"

    conf = OmegaConf.create({"out": "${strftime:[2025-06-01], %F_%H-%M-%S}"})
    assert conf["out"] == f"{expected}_00-00-00"
    conf2 = OmegaConf.create({"out": "${strftime:[2025-06-01], %F_%H-%M-%S, %F}"})
    assert conf2["out"] == expected


def test_listconfig_datetime():
    expected = "2025-06-01_12-30-00"

    conf = OmegaConf.create({"out": "${strftime:[2025-06-01T12:30:00], %F_%H-%M-%S}"})
    assert conf["out"] == expected
    conf2 = OmegaConf.create({"out": "${strftime:[2025-06-01T12:30:00], %F_%H-%M-%S, %F}"})
    assert conf2["out"] == expected


def test_dictconfig_date():
    expected = "2025-06-01"

    conf = OmegaConf.create({"out": "${strftime:{dt: 2025-06-01}, %F_%H-%M-%S}"})
    assert conf["out"] == f"{expected}_00-00-00"
    conf2 = OmegaConf.create({"out": "${strftime:{dt: 2025-06-01}, %F_%H-%M-%S, %F}"})
    assert conf2["out"] == expected


def test_dictconfig_datetime():
    expected = "2025-06-01_12-30-00"

    conf = OmegaConf.create({"out": "${strftime:{dt: 2025-06-01T12:30}, %F_%H-%M-%S}"})
    assert conf["out"] == expected
    conf2 = OmegaConf.create({"out": "${strftime:{dt: 2025-06-01T12:30}, %F_%H-%M-%S, %F}"})
    assert conf2["out"] == expected


def test_timezone():
    expected = "2025-06-01_12-30-00_-0500"

    conf = OmegaConf.create({"out": "${strftime:2025-06-01T12:30-0500, %F_%H-%M-%S_%z}"})
    assert conf["out"] == expected
    conf2 = OmegaConf.create({"out": "${strftime:2025-06-01T12:30-0500, %F_%H-%M-%S_%z, %F}"})
    assert conf2["out"] == expected

    expected = "2025-06-01_12-30-00_UTC"

    conf = OmegaConf.create({"out": "${strftime:2025-06-01T12:30Z, %F_%H-%M-%S_%Z}"})
    assert conf["out"] == expected
    conf2 = OmegaConf.create({"out": "${strftime:2025-06-01T12:30Z, %F_%H-%M-%S_%Z, %F}"})
    assert conf2["out"] == expected


@patch.object(sys, "argv", ["test_script.py", "param1=value1", "param2=value2"])
def test_param_resolver_all():
    expected = {"param1": "value1", "param2": "value2"}

    conf = OmegaConf.create({"out": "${param:}"})
    assert conf["out"] == expected


@patch.object(sys, "argv", ["test_script.py", "param1=value1"])
def test_param_resolver_single():
    expected = "value1"
    conf = OmegaConf.create({"out": "${param:param1}"})
    assert conf["out"] == expected


@patch.object(sys, "argv", ["test_script.py"])
def test_param_resolver_no_params():
    conf = OmegaConf.create({"out": "${param:param1}"})
    assert conf["out"] == ""


def test_register_alias():
    conf = OmegaConf.create({"out": "${cc.cmd:echo foo}"})
    assert conf["out"] == "foo"


def test_register_registered():
    register_omegaconf_resolver("cc.cmd", lambda x: x.split(" ")[0], replace=False)
    conf = OmegaConf.create({"out": "${cc.cmd:echo foo}"})
    assert conf["out"] == "foo"

    # Now replace
    register_omegaconf_resolver("cc.cmd", lambda x: x.split(" ")[0], replace=True)
    conf = OmegaConf.create({"out": "${cc.cmd:echo foo}"})
    assert conf["out"] == "echo"
