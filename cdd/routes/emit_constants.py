"""
Constant strings and tuples of strings which are to be interpolated in `emit.py`
"""
from string import Template

from cdd.pure_utils import indent_all_but_first

_create_route_desc = indent_all_but_first(
    '''"""Create `{name}`

```yml
responses:
  '201':
    description: A `{name}` object.
    content:
      application/json:
        schema:
          $ref: ```{name}```
  '400':
    description: A `ServerError` object.
    content:
      application/json:
        schema:
          $ref: ```ServerError```
```

:returns: Created ```{name}``` instance (as a dict), or an error
:rtype: ```dict```
"""'''
)
_create_helper_desc = "Create / handle-errors with Bottle and SQLalchemy"

create_route_variants = tuple(
    map(
        lambda s: Template(s).substitute(_create_route_desc=_create_route_desc),
        (
            """@{app}.post({route!r})
def create():
    $_create_route_desc
    try:
        config = {name}(**request.json)
    except TypeError as e:
        response.status = 400
        return {{"error": "ValidationError", "error_description": "\\n".join(e.args)}}

    try:
        with Session(engine) as session:
            session.add(config)
            session.commit()
            created = orm_to_dict(config)
    except DatabaseError as e:
        response.status = 400
        return {{"error": e.__class__.__name__, "error_code": e.code, "error_description": str(e.__cause__)}}

    return created
""",
            """@{app}.post({route!r})
def create():
    $_create_route_desc
    return create_helper0(request, response)({name})
""",
            """@{app}.post({route!r})
def create():
    $_create_route_desc
    code, body = create_helper1({name}, request.body)
    response.status = code
    return body
""",
        ),
    )
)

create_helper_variants = tuple(
    map(
        lambda s: Template(s).substitute(_create_helper_desc=_create_helper_desc),
        (
            '''def create_helper0(req, res):
    """
    $_create_helper_desc

    :param req: Bottle request
    :type req: ```bottle.request```

    :param res: Bottle response
    :type res: ```bottle.response```

    :returns: A function which actually does the work
    :rtype: ```Callable[[Base], dict]```
    """
    def _create_helper(orm_class):
        """
        $_create_helper_desc

        :param orm_class: An ORM class inheriting SQLalchemy declarative base class
        :type orm_class: ```Base```

        :returns: Created (as a dict) or error dict
        :rtype: ```dict```
        """
        try:
            orm_instance = orm_class(**req.json)
        except TypeError as e:
            res.status = 400
            return {{"error": "ValidationError", "error_description": "\\n".join(e.args)}}

        try:
            with Session(engine) as session:
                session.add(orm_instance)
                session.commit()
                created = orm_to_dict(orm_instance)
        except DatabaseError as e:
            res.status = 400
            return {{"error": e.__class__.__name__, "error_code": e.code, "error_description": str(e.__cause__)}}

        return created

    return _create_helper
''',
            '''def create_helper1(orm_class, body):
    """
    $_create_helper_desc

    :param orm_class: An ORM class inheriting SQLalchemy declarative base class
    :type orm_class: ```Base```

    :param body: Body of the instance to create
    :type body: ```dict```

    :returns: Status code, created (as a dict) or error dict
    :rtype: ```Tuple[int, dict]```
    """
    try:
        orm_instance = orm_class(**body)
    except TypeError as e:
        return 400, {{"error": "ValidationError", "error_description": "\\n".join(e.args)}}

    try:
        with Session(engine) as session:
            session.add(orm_instance)
            session.commit()
            created = orm_to_dict(orm_instance)
    except DatabaseError as e:
        return 400, {{"error": e.__class__.__name__, "error_code": e.code, "error_description": str(e.__cause__)}}

    return 201, created
''',
        ),
    )
)


read_route_variants = (
    '''
@{app}.get("{route}/:{id}")
def read({id}):
    """
    Find one `{name}` or error

    ```yml
    responses:
      '200':
        description: A `{name}` object.
        content:
          application/json:
            schema:
              $ref: ```{name}```
      '404':
        description: A `ServerError` object.
        content:
          application/json:
            schema:
              $ref: ```ServerError```
    ```

    :param {id}: The primary key of `{name}`
    :type {id}: ```str```

    :returns: Found `{name}` (as a dict) or error dict
    :rtype: ```dict```
    """
    with Session(engine) as session:
        config = session.execute(select({name}).filter_by({id}={id})).one_or_none()

    if config is None:
        response.status = 404
        return {{"error": "NotFound", "error_description": "{name} not found"}}

    return orm_to_dict(config[0])
''',
)

delete_route_variants = (
    '''
@{app}.delete("{route}/:{id}")
def destroy({id}):
    """
    Delete one `{name}`

    ```yml
    responses:
      '204':
    ```

    :param {id}: The primary key of `{name}`
    :type {id}: ```str```

    :returns: Found `{name}` (as a dict) or error dict
    :rtype: ```dict```
    """
    with Session(engine) as session:
        session.query({name}).filter({name}.{id} == {id}).delete(
            synchronize_session="evaluate"
        )

    response.status = 204
''',
)

__all__ = [
    "create_route_variants",
    "create_helper_variants",
    "read_route_variants",
    "delete_route_variants",
]
