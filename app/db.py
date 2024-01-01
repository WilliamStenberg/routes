from typing import List, Optional

import datetime
import os

from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool
from sqlalchemy.sql import func

import model as model
def engine():
    path = f'{os.getcwd()}/data/database.db'
    return create_engine(f'sqlite:///{path}', echo=True, poolclass=NullPool)

def setup():
    db_engine = engine()
    Base.metadata.create_all(db_engine)

def sess():
    return Session( engine())

class Base(DeclarativeBase):
    pass


class Map(Base):
    """
    Map of terrain where routes are run containing image and GPS coordinates

    Init fetches an image from OpenStreetView and stores it along with
    bounding box coordinates.
    """
    __tablename__ = "map"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[Optional[str]]
    mercator_bounding_box: Mapped["MercatorBoundingBox"] = relationship(
         back_populates="map", cascade="all, delete-orphan"
        )
    padded_route_bounding_box: Mapped["PaddedRouteBoundingBox"] = relationship(
         back_populates="map", cascade="all, delete-orphan"
        )

    image_path: Mapped[str]
    image_width: Mapped[int]
    image_height: Mapped[int]
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    routes: Mapped[List["Route"]] = relationship(
         back_populates="map", cascade="all, delete-orphan"
        )

def maps(sess):
    sess.execute(select(Map)).scalars()


def smallest_map_enclosing(sess, box) -> Optional[Map]:
    stmt = (
        select(Map)
        .join(PaddedRouteBoundingBox)
        .where(PaddedRouteBoundingBox.west <= box.west)
        .where(PaddedRouteBoundingBox.east >= box.east)
        .where(PaddedRouteBoundingBox.south <= box.south)
        .where(PaddedRouteBoundingBox.north >= box.north)
        .order_by(
((PaddedRouteBoundingBox.north - PaddedRouteBoundingBox.south)
*(PaddedRouteBoundingBox.east - PaddedRouteBoundingBox.west)).desc()
        )
    )
    sess.execute(stmt).scalars().first()


class Route(Base):
    """
    Route representation with string metadata and aggregated key numbers,
    reference to file and indexed by a lat/long point.
    """
    __tablename__ = "running_route"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str]
    file_name: Mapped[str]
    distance: Mapped[float]
    start_lat: Mapped[float]
    start_long: Mapped[float]

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    bounding_box: Mapped["RouteBoundingBox"] = relationship(
         back_populates="route", cascade="all, delete-orphan"
        )
    map_id: Mapped[int] = mapped_column(ForeignKey("map.id"))
    map: Mapped["Map"] = relationship(back_populates="routes")

def routes(sess) -> List[Route]:
    stmt = select(Route).order_by(Route.created_at.desc())
    res = sess.execute(stmt).scalars()
    return list(res)


class RouteBoundingBox(Base):
    __tablename__ = "route_bounding_box"
    id: Mapped[int] = mapped_column(primary_key=True)
    north: Mapped[float]
    east: Mapped[float]
    south: Mapped[float]
    west: Mapped[float]
    route_id: Mapped[int] = mapped_column(ForeignKey("running_route.id"))
    route: Mapped["Route"] = relationship(back_populates="bounding_box", single_parent=True)

    def __init__(self, box: model.BoundingBox):
        self.north=box.north
        self.east=box.east
        self.south=box.south
        self.west=box.west

    def bounding_box(self) -> model.BoundingBox:
        return model.BoundingBox(north=self.north, east=self.east, south=self.south, west=self.west)

class PaddedRouteBoundingBox(Base):
    __tablename__ = "padded_route_bounding_box"
    id: Mapped[int] = mapped_column(primary_key=True)
    north: Mapped[float]
    east: Mapped[float]
    south: Mapped[float]
    west: Mapped[float]
    map_id: Mapped[int] = mapped_column(ForeignKey("map.id"))
    map: Mapped["Map"] = relationship(back_populates="padded_route_bounding_box", single_parent=True)

    def __init__(self, box: model.BoundingBox):
        self.north=box.north
        self.east=box.east
        self.south=box.south
        self.west=box.west

    def bounding_box(self) -> model.BoundingBox:
        return model.BoundingBox(north=self.north, east=self.east, south=self.south, west=self.west)

class MercatorBoundingBox(Base):
    __tablename__ = "mercator_bounding_box"
    id: Mapped[int] = mapped_column(primary_key=True)
    north: Mapped[float]
    east: Mapped[float]
    south: Mapped[float]
    west: Mapped[float]
    map_id: Mapped[int] = mapped_column(ForeignKey("map.id"))
    map: Mapped["Map"] = relationship(back_populates="mercator_bounding_box", single_parent=True)
