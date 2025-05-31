from sqlalchemy import MetaData, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column

from core.settings import settings
from core.utils.case_converter import camel_case_to_snake_case

# создаем базовый класс для всех остальных
class Base(DeclarativeBase):
    __abstract__ = True

    created: Mapped[DateTime] = mapped_column(DateTime, default=func.now())
    updated: Mapped[DateTime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    metadata = MetaData(
        naming_convention=settings.database.naming_convention,
    )

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return f"{camel_case_to_snake_case(cls.__name__)}s"
