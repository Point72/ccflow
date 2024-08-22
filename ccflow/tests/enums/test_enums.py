import enum
import importlib
from unittest import TestCase
from unittest.mock import MagicMock, patch


class TestEnumType(object):
    def auto():
        pass


class TestEnum(TestCase):
    def tearDown(self) -> None:
        # Because test_init_parent and test_init_parent_csp muck around with imports
        # Make sure we always rest the imports at the end of each test so that other
        # tests are unaffected
        importlib.invalidate_caches()
        import ccflow.enums

        importlib.reload(ccflow.enums)

    def test_init_parent_no_csp(self):
        """Test initialization of ccflow.enum.Enum when csp is not importable."""

        import builtins

        original_import = builtins.__import__

        def no_csp_import(name, globals, locals, fromlist, level):
            if "csp" in name:
                raise ImportError
            else:
                return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = no_csp_import

        def f():
            import csp  # noqa: F401  # We use this to determine which path the test should take

        def g():
            from csp.impl.enum import Enum as BaseEnum  # noqa: F401

        try:
            self.assertRaises(ImportError, f)
            self.assertRaises(ImportError, g)

            import ccflow

            importlib.reload(ccflow.enums)

            from ccflow.enums import Enum

            self.assertTrue(issubclass(Enum, enum.Enum))

        finally:
            builtins.__import__ = original_import

    def test_init_parent_csp(self):
        """Test initialization of ccflow.enum.Enum when csp is importable."""
        mock_csp = MagicMock()
        mock_csp.Enum = TestEnumType
        mock_csp.DynamicEnum = MagicMock()
        mock_csp.EnumMeta = MagicMock()
        with patch.dict(
            "sys.modules",
            {"csp": mock_csp, "csp.impl": mock_csp, "csp.impl.enum": mock_csp},
        ):
            importlib.invalidate_caches()

            import csp

            import ccflow.enums

            importlib.reload(ccflow.enums)

            from ccflow.enums import Enum

            self.assertTrue(issubclass(Enum, csp.Enum))

    def test_init(self):
        """Test standard enum behavior. This test should pass in all environments (both with and without csp).
        No other enum functionality should be depended-upon.
        """
        from ccflow.enums import Enum, auto, make_enum

        class MyEnum(Enum):
            A = 0
            B = 1

        self.assertEqual(MyEnum.A.value, 0)
        self.assertEqual(MyEnum(1).name, "B")
        self.assertEqual(list(MyEnum), [MyEnum.A, MyEnum.B])
        self.assertTrue(issubclass(MyEnum, Enum))
        self.assertTrue(isinstance(MyEnum.A.name, str))
        self.assertEqual(MyEnum[MyEnum.A.name], MyEnum.A)

        class MyAutoEnum(Enum):
            A = auto()
            B = auto()

        self.assertEqual(MyAutoEnum.A.value, 0)
        self.assertEqual(list(MyAutoEnum), [MyAutoEnum.A, MyAutoEnum.B])
        self.assertTrue(issubclass(MyAutoEnum, Enum))

        MyDynamicEnum = make_enum("MyDynamicEnum", ["A", "B"], start=2)
        self.assertEqual(MyDynamicEnum.A.value, 2)
        self.assertEqual(list(MyDynamicEnum), [MyDynamicEnum.A, MyDynamicEnum.B])
        self.assertTrue(issubclass(MyDynamicEnum, Enum))
