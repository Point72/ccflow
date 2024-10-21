"""This module contains extension types for pydantic."""

from typing import Any

import jinja2
from pydantic import TypeAdapter
from pydantic_core import core_schema
from typing_extensions import Self


class JinjaTemplate(str):
    """String that is validated as a jinja2 template."""

    @property
    def template(self) -> jinja2.Template:
        """Return the underlying object that the path corresponds to."""
        return jinja2.Template(str(self))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_plain_validator_function(cls._validate)

    @classmethod
    def _validate(cls, value: Any) -> Self:
        if isinstance(value, JinjaTemplate):
            return value

        if isinstance(value, str):
            value = cls(value)
            try:
                value.template
            except Exception as e:
                raise ValueError(f"ensure this value contains a valid Jinja2 template string: {e}")

        return value

    @classmethod
    def validate(cls, value: Any) -> Self:
        """Try to convert/validate an arbitrary value to a JinjaTemplate."""
        return _TYPE_ADAPTER.validate_python(value)


_TYPE_ADAPTER = TypeAdapter(JinjaTemplate)
