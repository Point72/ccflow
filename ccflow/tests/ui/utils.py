"""Utility functions for UI tests."""

import panel as pn


def find_components_by_type(layout, component_type):
    """Recursively find all components of a given type in a Panel layout.

    Args:
        layout: A Panel layout or component to search
        component_type: The type of component to find

    Returns:
        A list of all components matching the given type
    """
    found = []
    if isinstance(layout, component_type):
        found.append(layout)
    if hasattr(layout, "objects"):
        for obj in layout.objects:
            found.extend(find_components_by_type(obj, component_type))
    if hasattr(layout, "__iter__") and not isinstance(layout, str):
        try:
            for item in layout:
                if hasattr(item, "objects") or isinstance(item, pn.viewable.Viewable):
                    found.extend(find_components_by_type(item, component_type))
        except TypeError:
            pass
    return found
