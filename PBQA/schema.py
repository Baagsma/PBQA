"""Engine-agnostic JSON schema preprocessing for structured generation.

These transformations apply regardless of which inference engine serves the
request; backends only decide where the resulting schema goes in the payload.
"""

import json

__all__ = ["resolve_refs", "lock_schema"]


def resolve_refs(schema: dict) -> dict:
    """Inline all $ref references in a JSON schema.

    Pydantic generates schemas with $ref pointers to $defs for enums, nested
    models, etc. Not all grammar engines resolve these correctly, which can
    cause enum fields to be unconstrained — the LLM then outputs objects
    instead of valid enum values.
    """
    schema = json.loads(json.dumps(schema))  # deep copy
    defs = schema.get("$defs", {})

    def _resolve(node):
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref_path = node["$ref"]  # e.g. "#/$defs/Difficulty"
            if ref_path.startswith("#/$defs/"):
                def_name = ref_path[len("#/$defs/"):]
                if def_name in defs:
                    resolved = json.loads(json.dumps(defs[def_name]))
                    return _resolve(resolved)
            return node
        return {k: _resolve_value(v) for k, v in node.items()}

    def _resolve_value(value):
        if isinstance(value, dict):
            return _resolve(value)
        if isinstance(value, list):
            return [_resolve_value(item) for item in value]
        return value

    resolved = _resolve(schema)
    resolved.pop("$defs", None)
    return resolved


def lock_schema(schema: dict) -> dict:
    """Set additionalProperties: false on all object types in a JSON schema.

    This is required for grammar-constrained generation engines (e.g. llguidance)
    that default additionalProperties to true per the JSON Schema spec, which
    allows the model to output arbitrary extra keys.
    """
    schema = json.loads(json.dumps(schema))  # deep copy
    defs = schema.get("$defs", {})

    def _lock(node):
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
        for value in node.values():
            if isinstance(value, dict):
                _lock(value)
            elif isinstance(value, list):
                for item in value:
                    _lock(item)

    _lock(schema)
    for defn in defs.values():
        _lock(defn)

    return schema
