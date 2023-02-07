from sqlalchemy import Boolean, Column, Float, Identity, Integer, String
from sqlalchemy.orm import declarative_base
Base = declarative_base()

class example(Base):
    __tablename__ = 'example'
    one = Column(Float)
    two = Column(String)
    three = Column(Boolean)
    __index_level_0__ = Column(String)
    id = Column(Integer, primary_key=True, server_default=Identity())

    def __repr__(self):
        """
        Emit a string representation of the current instance
        
        :return: String representation of instance
        :rtype: ```str```
        """
        return (
            'example(one={one!r}, two={two!r}, three={three!r}, __index_level_0__={__index_level_0__!r}, id={id!r})'
            .format(one=self.one, two=self.two, three=self.three,
            __index_level_0__=self.__index_level_0__, id=self.id))


__all__ = ['example']
