from datetime import date
from pathlib import Path
from socket import gethostname

from omegaconf import OmegaConf

import ccflow.plugins.omegaconf_resolvers  # noqa: F401


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
