"""This module contains extension types for pydantic."""

from typing import Any

import jinja2


class JinjaTemplate(str):
    """String that is validated as a jinja2 template."""

    @property
    def template(self) -> jinja2.Template:
        """Return the underlying object that the path corresponds to."""
        return jinja2.Template(str(self))

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value, field=None) -> Any:
        if isinstance(value, JinjaTemplate):
            return value

        if isinstance(value, str):
            value = cls(value)
            try:
                value.template
            except Exception as e:
                raise ValueError(f"ensure this value contains a valid Jinja2 template string: {e}")

        return value
