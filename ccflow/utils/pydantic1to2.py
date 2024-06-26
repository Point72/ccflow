import pydantic
from packaging import version

if version.parse(pydantic.__version__) < version.parse("2"):
    import pydantic.errors as errors  # noqa F401, F403
    import pydantic.typing as typing  # noqa F401, F403
    from pydantic import *  # noqa F401, F403
    from pydantic import ValidationError
    from pydantic.generics import GenericModel

    ValidationTypeError = ValidationError
else:
    import pydantic.v1.errors as errors  # noqa F401, F403
    import pydantic.v1.typing as typing  # noqa F401, F403
    from pydantic import ValidationError
    from pydantic.v1 import *  # noqa F401, F403

    # https://docs.pydantic.dev/latest/errors/errors/
    ValidationTypeError = ValueError

    class GenericModel:
        """Dummy GenericModel class for pydantic 2 compatibility."""

        pass


def annotation(field):
    if version.parse(pydantic.__version__) < version.parse("2"):
        return field.annotation
    else:
        return field.type_
