from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

# So we can register the resolvers
import ccflow.plugins  # noqa: F401


class TestTodayResolver:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        """
        Setup mock at class level using the correct import path
        """
        fixed_dt = datetime(2024, 1, 1, 15, 0, 0, tzinfo=ZoneInfo("UTC"))
        # Patch the correct module path
        patcher = patch("ccflow.plugins.omegaconf_resolvers.datetime", autospec=True)
        mock_dt = patcher.start()
        mock_dt.now.return_value = fixed_dt
        yield
        patcher.stop()

    def test_basic_utc(self):
        """Test UTC date resolution"""
        cfg = OmegaConf.create({"date": "${today_at_tz:UTC}"})
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["date"] == date(2024, 1, 1)

    def test_sydney_date(self):
        """Test Sydney date (should be next day due to timezone)"""
        cfg = OmegaConf.create({"date": "${today_at_tz:Australia/Sydney}"})
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["date"] == date(2024, 1, 2)  # Next day in Sydney

    def test_new_york_date(self):
        """Test New York date (should be same day)"""
        cfg = OmegaConf.create({"date": "${today_at_tz:America/New_York}"})
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["date"] == date(2024, 1, 1)  # Same day in NY

    def test_invalid_timezone(self):
        """Test invalid timezone raises error"""
        cfg = OmegaConf.create({"date": "${today_at_tz:Invalid/Timezone}"})
        with pytest.raises(InterpolationResolutionError):
            OmegaConf.to_container(cfg, resolve=True)

    def test_no_timezone(self):
        """Test invalid timezone raises error"""
        cfg = OmegaConf.create({"date": "${today_at_tz:}"})
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["date"] == date(2024, 1, 1)

        cfg = OmegaConf.create({"date": "${today_at_tz:null}"})
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["date"] == date(2024, 1, 1)

    def test_multiple_timezones(self):
        """Test multiple timezone dates in same config"""
        cfg = OmegaConf.create({"sydney": "${today_at_tz:Australia/Sydney}", "ny": "${today_at_tz:America/New_York}", "utc": "${today_at_tz:UTC}"})
        resolved = OmegaConf.to_container(cfg, resolve=True)

        assert resolved["sydney"] == date(2024, 1, 2)  # Next day
        assert resolved["ny"] == date(2024, 1, 1)  # Same day
        assert resolved["utc"] == date(2024, 1, 1)  # Reference day


class TestIsMissingResolver:
    def test_missing_value(self):
        """Test when value is missing from parent"""
        cfg = OmegaConf.create({"exists": "value", "should_be_true": "${is_missing:missing_key}"})
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["should_be_true"] is True

    def test_existing_value(self):
        """Test when value exists in parent"""
        cfg = OmegaConf.create({"exists": "value", "should_be_false": "${is_missing:exists}"})
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["should_be_false"] is False

    def test_nested_config(self):
        """Test with nested configuration"""
        cfg = OmegaConf.create({"nested": {"exists": "value"}, "checks": {"missing": "${is_missing:not_here}", "exists": "${is_missing:exists}"}})
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["checks"]["missing"] is True  # it is missing
        assert resolved["checks"]["exists"] is False  # it is not missing

    def test_practical_usage(self):
        """Test practical usage for module enabling/disabling"""
        cfg = OmegaConf.create(
            {
                "input_data": "exists",
                "module1": {
                    # The resolver looks at module1 scope, not root
                    "missing": "${is_missing:not_configured}"
                },
                "module2": {
                    # The resolver looks at module2 scope, not root
                    "missing": "${is_missing:input_data}"
                },
                "module3": {
                    # The resolver looks at module3 scope, not root
                    "input_data": "exists",
                    "missing": "${is_missing:input_data}",
                },
            }
        )
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["module1"]["missing"] is True
        assert resolved["module2"]["missing"] is True
        assert resolved["module3"]["missing"] is False


class TestListToStaticDictResolver:
    def test_basic_conversion(self):
        """Test basic list to static dictionary conversion"""
        config = OmegaConf.create(
            {
                "input_list": ["key1", "key2"],
                "static_elements": {"static_elt1": "a", "static_elt2": 10},
                "result_dict": "${list_to_static_dict:${input_list},${static_elements}}",
            }
        )

        results = OmegaConf.to_container(config, resolve=True)
        assert results == {
            "input_list": ["key1", "key2"],
            "static_elements": {"static_elt1": "a", "static_elt2": 10},
            "result_dict": {"key1": {"static_elt1": "a", "static_elt2": 10}, "key2": {"static_elt1": "a", "static_elt2": 10}},
        }


class TestDictFromTuplesResolver:
    def test_basic_conversion(self):
        """Test basic tuple to dictionary conversion"""
        config = OmegaConf.create(
            {
                "dict_key": "Key",
                "dict_value": "Value",
                "result_dict": "${dict_from_tuples:[[${dict_key},${dict_value}],[StaticKey,StaticValue]]}",
            }
        )

        results = OmegaConf.to_container(config, resolve=True)
        assert results == {
            "dict_key": "Key",
            "dict_value": "Value",
            "result_dict": {"Key": "Value", "StaticKey": "StaticValue"},
        }

    def test_empty_tuple(self):
        """Test that empty tuples raise an error"""
        config = OmegaConf.create({"result_dict": "${dict_from_tuples:[[]]}"})

        with pytest.raises(InterpolationResolutionError) as exc_info:
            OmegaConf.to_container(config, resolve=True)
        assert "not enough values to unpack (expected 2, got 0)" in str(exc_info.value)

    def test_single_element_tuple(self):
        """Test that single element tuples raise an error"""
        config = OmegaConf.create({"result_dict": "${dict_from_tuples:[[SingleElement]]}"})

        with pytest.raises(InterpolationResolutionError) as exc_info:
            OmegaConf.to_container(config, resolve=True)
        assert "not enough values to unpack (expected 2, got 1)" in str(exc_info.value)

    def test_three_element_tuple(self):
        """Test that three element tuples raise an error"""
        config = OmegaConf.create({"result_dict": "${dict_from_tuples:[[One,Two,Three]]}"})

        with pytest.raises(InterpolationResolutionError) as exc_info:
            OmegaConf.to_container(config, resolve=True)
        assert "too many values to unpack (expected 2)" in str(exc_info.value)


class TestTrimNullValuesResolver:
    def test_basic_trim(self):
        """Test trimming null values from dictionary"""
        config = OmegaConf.create({"input_dict": {"k1": "v1", "k2": "v2", "k3": None}, "result_dict": "${trim_null_values:${input_dict}}"})

        results = OmegaConf.to_container(config, resolve=True)
        assert results == {
            "input_dict": {"k1": "v1", "k2": "v2", "k3": None},
            "result_dict": {
                "k1": "v1",
                "k2": "v2",
            },
        }


class TestStringReplaceResolver:
    def test_basic_replace(self):
        """Test basic string replacement"""
        config = OmegaConf.create({"input_val": "TestA", "result_val": "${replace:${input_val},A,B}", "untouched": True})

        results = OmegaConf.to_container(config, resolve=True)
        assert results == {"input_val": "TestA", "result_val": "TestB", "untouched": True}


class TestIsNoneOrEmptyResolver:
    def test_various_values(self):
        """Test none/empty detection for various values"""
        config = OmegaConf.create(
            {
                "null_value": None,
                "empty_string": "",
                "is_value_none": "${is_none_or_empty:${null_value}}",
                "is_empty_string": "${is_none_or_empty:${empty_string}}",
                "is_not_null": "${is_none_or_empty:Value}",
            }
        )

        results = OmegaConf.to_container(config, resolve=True)
        assert results == {
            "null_value": None,
            "empty_string": "",
            "is_value_none": True,
            "is_empty_string": True,
            "is_not_null": False,
        }


class TestIsNotResolver:
    def test_boolean_negation(self):
        """Test boolean value negation"""
        config = OmegaConf.create(
            {
                "bool_true": True,
                "bool_false": False,
                "result_false": "${is_not:${bool_true}}",
                "result_true": "${is_not:${bool_false}}",
            }
        )

        results = OmegaConf.to_container(config, resolve=True)
        assert results == {
            "bool_true": True,
            "bool_false": False,
            "result_false": False,
            "result_true": True,
        }


class TestIfElseResolver:
    def test_conditional_values(self):
        """Test conditional value selection"""
        config = OmegaConf.create(
            {
                "bool_true": True,
                "bool_false": False,
                "value_true": "Value if true",
                "value_false": "Value if false",
                "result_false": "${if_else:${bool_false},${value_true},${value_false}}",
                "result_true": "${if_else:${bool_true},${value_true},${value_false}}",
            }
        )

        results = OmegaConf.to_container(config, resolve=True)
        assert results == {
            "bool_true": True,
            "bool_false": False,
            "value_true": "Value if true",
            "value_false": "Value if false",
            "result_false": "Value if false",
            "result_true": "Value if true",
        }
