"""Helpers for inspecting the serialized spaday component tree in tests."""


def iter_nodes(node):
    """Yield ``node`` and every descendant node (depth-first) of a ``to_node()`` dict."""
    yield node
    for children in node.get("slots", {}).values():
        for child in children:
            yield from iter_nodes(child)


def nodes_with_tag(node, tag):
    """All nodes in the tree with the given element ``tag``."""
    return [n for n in iter_nodes(node) if n.get("tag") == tag]


def text_of(node):
    """The node's ``textContent`` string, or None."""
    tc = node.get("props", {}).get("textContent")
    return tc.get("Str") if isinstance(tc, dict) else None


def all_text(node):
    """Every ``textContent`` string found in the tree."""
    return [t for t in (text_of(n) for n in iter_nodes(node)) if t is not None]


def prop_str(node, name):
    """A node prop serialized as a string (the ``{"Str": value}`` tag), or None."""
    value = node.get("props", {}).get(name)
    return value.get("Str") if isinstance(value, dict) else None


def click_set_field(node):
    """The literal value written by a ``click`` SetField action on the node, or None."""
    event = node.get("events", {}).get("click")
    if event and event.get("kind") == "set-field":
        return event["value"]["value"]
    return None


def show_when_value(node):
    """The literal a ``spa-show`` compares ``selected`` against in its ``when`` binding, or None."""
    when = node.get("bindings", {}).get("when")
    if not when or "compute" not in when:
        return None
    expr = when["compute"]
    if expr.get("expr") == "eq":
        return expr["b"].get("value")
    return None
