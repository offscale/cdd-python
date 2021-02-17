"""
Emit constant strings with interpolated values for route generation
"""
from cdd.routes.emit_constants import (
    create_helper_variants,
    create_route_variants,
    delete_route_variants,
    read_route_variants,
)


def create(app, name, route, variant=2):
    """
    Create the `create` route

    :param app: Variable name (Bottle App)
    :type app: ```str```

    :param name: Name of entity
    :type name: ```str```

    :param route: The path of the resource
    :type route: ```str```

    :param variant: Number of variant
    :type variant: ```int```

    :returns: Create route variant with interpolated values
    :rtype: ```str```
    """
    return create_route_variants[variant].format(app=app, name=name, route=route)


def create_util(name, route, variant=1):
    """
    Create utility function that the `create` emitter above uses

    :param name: Name of entity
    :type name: ```str```

    :param route: The path of the resource
    :type route: ```str```

    :param variant: Number of variant
    :type variant: ```int```

    :returns: Create route variant with interpolated values
    :rtype: ```str```
    """
    return create_helper_variants[variant].format(name=name, route=route)


def read(app, name, route, primary_key, variant=0):
    """
    Create the `read` route

    :param app: Variable name (Bottle App)
    :type app: ```str```

    :param name: Name of entity
    :type name: ```str```

    :param route: The path of the resource
    :type route: ```str```

    :param primary_key: The id
    :type primary_key: ```Any```

    :param variant: Number of variant
    :type variant: ```int```

    :returns: Create route variant with interpolated values
    :rtype: ```str```
    """
    return read_route_variants[variant].format(
        app=app, name=name, route=route, id=primary_key
    )


def destroy(app, name, route, primary_key, variant=0):
    """
    Create the `destroy` route

    :param app: Variable name (Bottle App)
    :type app: ```str```

    :param name: Name of entity
    :type name: ```str```

    :param route: The path of the resource
    :type route: ```str```

    :param primary_key: The id
    :type primary_key: ```Any```

    :param variant: Number of variant
    :type variant: ```int```

    :returns: Create route variant with interpolated values
    :rtype: ```str```
    """
    return delete_route_variants[variant].format(
        app=app, name=name, route=route, id=primary_key
    )


__all__ = ["create", "create_util", "read", "destroy"]
