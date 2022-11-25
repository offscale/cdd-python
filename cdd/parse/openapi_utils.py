"""
OpenAPI parser utility functions
"""


def extract_entities(openapi_str):
    """
    Extract entities from an OpenAPI string, where entities are defines as anything within "```"

    :param openapi_str: The OpenAPI str
    :type openapi_str: ```str```

    :return: Entities
    :rtype: ```List[str]```
    """
    entities, ticks, space, stack = [], 0, 0, []

    def add_then_clear_stack():
        """
        Join entity, if non empty add to entities. Clear stack.
        """
        entity = "".join(stack)
        if entity:
            entities.append(entity)
        stack.clear()

    for idx, ch in enumerate(openapi_str):
        if ch.isspace():
            space += 1
            add_then_clear_stack()
            ticks = 0
        elif ticks > 2:
            ticks, space = 0, 0
            stack and add_then_clear_stack()
            stack.append(ch)
        elif ch == "`":
            ticks += 1
        elif stack:
            stack.append(ch)
    add_then_clear_stack()
    return entities


__all__ = ["extract_entities"]
