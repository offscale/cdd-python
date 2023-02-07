config_tbl = Table('example', metadata, Column('one', Float), Column('two',
    String), Column('three', Boolean), Column('__index_level_0__', String),
    Column('id', Integer, primary_key=True, server_default=Identity()))
__all__ = ['example']
