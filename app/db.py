from typing import List, Optional, Tuple

import datetime
import os
from PIL import Image

from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

import model as model
import maps as maps
import utils as utils

def engine():
    path = f'{os.getcwd()}/data/database.db'
    return create_engine(f'sqlite:///{path}', echo=True)

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

def ensure_persistent_map(sess, ts: model.Timeseries) -> Tuple[Map, model.Map]:
    tight_box = ts.bounding_box()
    padded_rect = tight_box.padded_rect()
    found_map = smallest_map_enclosing(sess, tight_box)
    twice_padded = padded_rect.padded_rect()
    # create a new map if we
    # a) didn't find a map enclosing the tight box, or
    # b) found a too large map, meaning that twice_padded does not contain the found_map's padded box
    if found_map is None or not twice_padded.contains(found_map.padded_route_bounding_box.bounding_box()):
        file_path = utils.IMAGEPATH + str(padded_rect) + '.png'
        m = maps.transient_map(padded_rect)
        m.image.save(file_path)
        new_map = Map(
            padded_route_bounding_box = PaddedRouteBoundingBox(padded_rect),
            mercator_bounding_box = MercatorBoundingBox(m.mercator_box),
            image_path=file_path,
            image_width=m.image.width,
            image_height=m.image.height
            )
        sess.add(new_map)
        return new_map, m
    else:
        return found_map, model.Map(
            box=found_map.padded_route_bounding_box.bounding_box(),
            mercator_box=found_map.mercator_bounding_box.box(),
            image=Image.open(found_map.image_path))


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
*(PaddedRouteBoundingBox.east - PaddedRouteBoundingBox.west))
        )
    )
    return sess.execute(stmt).scalars().first()


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

    def __init__(self, box: model.MercatorBox):
        self.north=box.north
        self.east=box.east
        self.south=box.south
        self.west=box.west

    def box(self) -> model.MercatorBox:
        return model.MercatorBox(north=self.north, east=self.east, south=self.south, west=self.west)
