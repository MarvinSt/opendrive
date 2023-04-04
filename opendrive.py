import os
import json
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np

from geometry import Geometry, Poly3, RotateX, RotateY, RotateZ, get_normal
from dataclasses import dataclass


@dataclass
class Speed():
    max: np.float64
    unit: Optional[str] = None


@dataclass
class RoadLinkElement():
    elementId: str
    contactPoint: Optional[str] = None
    elementDir: Optional[str] = None
    elementS: Optional[str] = None
    elementType: Optional[str] = None


class RoadLink():
    predecessor: list[RoadLinkElement]
    successor: list[RoadLinkElement]

    def __init__(self):
        self.predecessor = []
        self.successor = []
        pass

    @staticmethod
    def parse(node):
        rl = RoadLink()
        for element in node:
            el_tag = element.tag
            if el_tag == "predecessor":
                rl.predecessor.append(RoadLinkElement(**element.attrib))
            elif el_tag == "successor":
                rl.successor.append(RoadLinkElement(**element.attrib))

        return rl


class RoadType():
    speed: list[Speed]

    s: np.float64
    type: str
    country: str

    def __init__(self, s, type, country=""):
        self.s = s
        self.type = type
        self.country = country

        self.speed = []

    @staticmethod
    def parse(node):
        rt = RoadType(**node.attrib)
        for element in node:
            rt.speed.append(Speed(**element.attrib))
        return rt


class RoadElevationProfile():
    elevation: list[Poly3]

    def __init__(self):
        self.elevation = []

    def get_height_evelation(self, s):
        assert s >= np.float64(0), "s should be larger or zero"
        for elevation in self.elevation[::-1]:
            if s >= elevation.s:
                return elevation.eval_s(s)
        return np.float64(0)

    @staticmethod
    def parse(node):
        re = RoadElevationProfile()
        for element in node:
            re.elevation.append(Poly3(**element.attrib))
        return re


class RoadLateralProfile():
    superelevation: list[Poly3]
    shape: list[Poly3]

    def __init__(self):
        self.superelevation = []
        self.shape = []

    def get_angle_superelevation(self, s):
        assert s >= np.float64(0), "s should be larger or zero"
        for superelevation in self.superelevation[::-1]:
            if s >= superelevation.s:
                return superelevation.eval_s(s)
        return np.float64(0)

    def get_height_shape(self, s, t):
        # TODO!
        return np.float(0)

    @staticmethod
    def parse(node):
        rlp = RoadLateralProfile()
        for element in node:
            el_tag = element.tag
            if el_tag == "superelevation":
                rlp.superelevation.append(Poly3(**element.attrib))
            elif el_tag == "shape":
                rlp.shape.append(Poly3(**element.attrib))

        return rlp


class RoadPlanView():
    geometry: list[Geometry]

    def __init__(self):
        self.geometry = []

    def eval_s(self, s):
        assert s >= np.float64(0), "s should be larger or zero"
        for g in self.geometry[::-1]:
            if s >= g.s:
                return g.get_xyt(s)

    @staticmethod
    def parse(node):
        pv = RoadPlanView(**node.attrib)
        for element in node:
            geom_type = element[0].tag
            geom_attrib = {**element.attrib, **element[0].attrib}

            geom = None
            if geom_type == "line":
                geom = Geometry.Line(**geom_attrib)
            elif geom_type == "arc":
                geom = Geometry.Arc(**geom_attrib)
            elif geom_type == "spiral":
                geom = Geometry.Spiral(**geom_attrib)
            else:
                print(f"Not implemented! Geometry {geom}")
            pv.geometry.append(geom)
        return pv


class LaneHeight():
    inner: np.float64
    outer: np.float64
    sOffset: np.float64

    def __init__(self, inner, outer, sOffset):
        self.inner = np.float64(inner)
        self.outer = np.float64(outer)
        self.sOffset = np.float64(sOffset)


class LaneLink():
    predecessor: list[np.int8]
    successor: list[np.int8]

    def __init__(self):
        self.predecessor = []
        self.successor = []

    @staticmethod
    def parse(node):
        rl = RoadLink()
        for element in node:
            el_tag = element.tag
            lane_id = np.int8(element.attrib['id'])
            if el_tag == "predecessor":
                rl.predecessor.append(lane_id)
            elif el_tag == "successor":
                rl.successor.append(lane_id)
        return rl


class RoadMark():
    color: str
    height: np.float64
    laneChange: str
    material: str
    sOffset: np.float64
    type: str
    weight: str
    width: np.float64

    def __init__(self, type, sOffset, color="standard", height=np.float64(0.01), material="standard", laneChange="both", weight=None, width=np.float64(0.1)):
        self.color = color
        self.height = np.float64(height)
        self.laneChange = laneChange
        self.material = material
        self.sOffset = np.float64(sOffset)
        self.type = type
        self.weight = weight
        self.width = np.float64(width)

    @staticmethod
    def parse(node):
        rm = RoadMark(**node.attrib)
        for element in node:
            # TODO: Implement detailed lane marking
            el_tag = element.tag
            print(f"Not implemented! RoadMark {el_tag}")

        return rm


class Lane():
    id: np.int8
    type: str
    level: bool

    border: list[Poly3]
    width: list[Poly3]
    height: list[LaneHeight]
    link: LaneLink
    roadmark: list[RoadMark]

    def __init__(self, id, type, level=False):
        self.id = np.int8(id)
        self.type = type
        self.level = level

        self.border: list[Poly3] = []
        self.width: list[Poly3] = []
        self.height: list[LaneHeight] = []
        self.roadmark: list[RoadMark] = []
        self.link: LaneLink = None

    def get_lane_border(self, s):
        assert s >= np.float64(0), "s should be larger or zero"
        for border in self.border[::-1]:
            if s >= border.s:
                return border.eval_s(s)
        return np.float64(0)

    def get_lane_width(self, s):
        assert s >= np.float64(0), "s should be larger or zero"
        for width in self.width[::-1]:
            if s >= width.s:
                return width.eval_s(s)
        return np.float64(0)

    def get_lane_height(self, s):
        assert s >= np.float64(0), "s should be larger or zero"
        for height in self.height[::-1]:
            if s >= height.sOffset:
                return height.inner, height.outer
        return np.float64(0), np.float64(0)

    def get_road_mark(self, s):
        assert s >= np.float64(0), "s should be larger or zero"
        for roadmark in self.roadmark[::-1]:
            if s >= roadmark.sOffset:
                return roadmark
        return None

    def has_border(self):
        return len(self.border) > 0

    @staticmethod
    def parse(node):
        ln = Lane(**node.attrib)
        for element in node:
            el_tag = element.tag
            if el_tag == "border":
                elem_attr = element.attrib
                elem_attr['s'] = elem_attr['sOffset']
                del elem_attr['sOffset']
                ln.border.append(Poly3(**elem_attr))
            elif el_tag == "width":
                elem_attr = element.attrib
                elem_attr['s'] = elem_attr['sOffset']
                del elem_attr['sOffset']
                ln.width.append(Poly3(**elem_attr))
            elif el_tag == "userData":
                # TODO: To implement
                pass
            elif el_tag == "roadMark":
                ln.roadmark.append(RoadMark.parse(element))
            elif el_tag == "link":
                ln.link = LaneLink.parse(element)
            elif el_tag == "height":
                ln.height.append(LaneHeight(**element.attrib))
            else:
                print(f"Not implemented! Lane {el_tag}")

        return ln

    """
    <xs:element name="link" type="t_road_lanes_laneSection_lcr_lane_link" minOccurs="0" maxOccurs="1"/>
    <xs:choice minOccurs="1" maxOccurs="unbounded">
        <xs:element name="border" type="t_road_lanes_laneSection_lr_lane_border" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="width" type="t_road_lanes_laneSection_lr_lane_width" minOccurs="0" maxOccurs="unbounded"/>
    </xs:choice>
    <xs:element name="roadMark" type="t_road_lanes_laneSection_lcr_lane_roadMark" minOccurs="0" maxOccurs="unbounded"/>
    <xs:element name="material" type="t_road_lanes_laneSection_lr_lane_material" minOccurs="0" maxOccurs="unbounded"/>
    <xs:element name="speed" type="t_road_lanes_laneSection_lr_lane_speed" minOccurs="0" maxOccurs="unbounded"/>
    <xs:element name="access" type="t_road_lanes_laneSection_lr_lane_access" minOccurs="0" maxOccurs="unbounded"/>
    <xs:element name="height" type="t_road_lanes_laneSection_lr_lane_height" minOccurs="0" maxOccurs="unbounded"/>
    <xs:element name="rule" type="t_road_lanes_laneSection_lr_lane_rule" minOccurs="0" maxOccurs="unbounded"/>
    """


class LaneSection():
    left: list[Lane]
    center: list[Lane]
    right: list[Lane]

    s: np.float64
    singleSide: bool

    def __init__(self, s, singleSide=False):
        self.left: list[Lane] = []
        self.center: list[Lane] = []
        self.right: list[Lane] = []

        self.s = np.float64(s)
        self.singleSide = singleSide == "true"

    @staticmethod
    def parse(node):
        ls = LaneSection(**node.attrib)
        for element in node:
            el_tag = element.tag
            if el_tag == "left":
                for ln in element:
                    ls.left.append(Lane.parse(ln))
            elif el_tag == "center":
                for ln in element:
                    ls.center.append(Lane.parse(ln))
            elif el_tag == "right":
                for ln in element:
                    ls.right.append(Lane.parse(ln))
        return ls


class Lanes():
    laneOffset: list[Poly3]
    laneSection: list[LaneSection]

    def __init__(self):
        self.laneOffset: list[Poly3] = []
        self.laneSection: list[LaneSection] = []

    def get_lane_offset(self, s):
        assert s >= np.float64(0), "s should be larger or zero"
        for laneoffset in self.laneOffset[::-1]:
            if s >= laneoffset.s:
                return laneoffset.eval_s(s)
        return np.float64(0)

    @staticmethod
    def parse(node):
        lns = Lanes()
        for element in node:
            el_tag = element.tag
            if el_tag == "laneOffset":
                pl3 = Poly3(**element.attrib)
                lns.laneOffset.append(pl3)
            elif el_tag == "laneSection":
                ls = LaneSection.parse(element)
                lns.laneSection.append(ls)
            else:
                print(f"Not implemented! Lanes {el_tag}")
        return lns


class Road():
    name: str
    length: np.float64
    id: str
    junction: str
    rule: str

    # <xs:element name="link" type="t_road_link" minOccurs="0" maxOccurs="1"/>
    # <xs:element name="type" type="t_road_type" minOccurs="0" maxOccurs="unbounded"/>
    # <xs:element name="planView" type="t_road_planView" minOccurs="1" maxOccurs="1"/>
    # <xs:element name="elevationProfile" type="t_road_elevationProfile" minOccurs="0" maxOccurs="1"/>
    # <xs:element name="lateralProfile" type="t_road_lateralProfile" minOccurs="0" maxOccurs="1"/>
    # <xs:element name="lanes" type="t_road_lanes" minOccurs="1" maxOccurs="1"/>
    # <xs:element name="objects" type="t_road_objects" minOccurs="0" maxOccurs="1"/>
    # <xs:element name="signals" type="t_road_signals" minOccurs="0" maxOccurs="1"/>
    # <xs:element name="surface" type="t_road_surface" minOccurs="0" maxOccurs="1"/>
    # <xs:element name="railroad" type="t_road_railroad" minOccurs="0" maxOccurs="1"/>

    planview: RoadPlanView
    elevationProfile: RoadElevationProfile = None
    lateralProfile: RoadLateralProfile = None
    type: RoadType
    link: RoadLink
    lanes: Lanes

    def __init__(self, length, id, junction, name="", rule=""):
        self.name = name
        self.length = np.float64(length)
        self.id = id
        self.junction = junction
        self.rule = rule

    def get_reference_line(self, s):
        # evaluate reference line, elevation and lateral profile
        x, y, hdg = self.planview.eval_s(s)

        z = self.elevationProfile.get_height_evelation(
            s) if self.elevationProfile else np.float64(0)

        a_roll = self.lateralProfile.get_angle_superelevation(
            s)if self.lateralProfile else np.float64(0)

        return x, y, z, hdg, a_roll

    @staticmethod
    def parse(node):
        rd = Road(**node.attrib)
        for element in node:
            el_tag = element.tag
            if el_tag == "planView":
                rd.planview = RoadPlanView.parse(element)
            elif el_tag == "elevationProfile":
                rd.elevationProfile = RoadElevationProfile.parse(element)
            elif el_tag == "lateralProfile":
                rd.lateralProfile = RoadLateralProfile.parse(element)
            elif el_tag == "type":
                rd.type = RoadType.parse(element)
            elif el_tag == "link":
                rd.link = RoadLink.parse(element)
            elif el_tag == "lanes":
                rd.lanes = Lanes.parse(element)
            elif el_tag == "objects":
                # TODO!
                pass
            elif el_tag == "signals":
                # TODO!
                pass
            else:
                print(f"Not implemented! Road {el_tag}")

        return rd


class OpenDrive():
    road: list[Road]

    def __init__(self):
        self.road = []

    def get_road_ref(self, s_step=1.0, road_index=-1):
        mesh = {}
        roads = self.road if road_index == -1 else [road[road_index]]

        for road in roads:
            mesh[road.id] = []

            s_int = np.float64(0)

            while True:
                # saturate the road length
                s_int = min(s_int, road.length)

                x, y, z, hdg, roll = road.get_reference_line(s_int)

                mesh[road.id].append(np.array([x, y, z, hdg, roll, s_int]))

                if s_int >= road.length:
                    break

                # increment step size
                s_int += s_step

        return mesh

    def get_lane_bounds(self, s, section: LaneSection, p_cen, th_vec, t_vec, l_offset, left_side=True):
        """
        NOTE: This logic is not yet foolproof. There are a couple short commings:
        * Duplicate coordinates are possible between neighbouring lanes
        * There is no consideration for continuity between neighboring lanes due to height differences, i.e. there could be gaps

        One possible solution is to have a secondary pass:
        * Loop over all neighbouring lanes and merge common vertices
        * Create a list of triangle faces with these known vertices
            while taking jumps in height into consideration
            this is potentially problematic if we have single sided sections
        * We also need to follow connecting lane sections and join these together
        """
        lane_mesh = {}

        # start at the center position, handle a lane offset here directly
        p_lane = p_cen.copy() + l_offset * t_vec

        sign = np.float64(1.0) if left_side else np.float64(-1.0)

        lanes = section.left[::-1] if left_side else section.right
        for lane in lanes:
            # get lane width and height (inner, outer)
            ds = max(s - section.s, np.float64(0.0))
            lane_width = sign * lane.get_lane_width(ds)
            h_inner, h_outer = lane.get_lane_height(ds)

            h_inner = np.float64(0)
            h_outer = np.float64(0)

            # calculate bound coordinates
            if lane.level:
                p_lane_inner = p_lane.copy() + \
                    h_inner * get_normal(th_vec)
                p_lane += lane_width * th_vec
                p_lane_outer = p_lane.copy() + \
                    h_outer * get_normal(th_vec)
            else:
                p_lane_inner = p_lane.copy() + \
                    h_inner * get_normal(t_vec)
                p_lane += lane_width * t_vec
                p_lane_outer = p_lane.copy() + \
                    h_outer * get_normal(t_vec)

            if lane.id not in lane_mesh:
                lane_mesh[lane.id] = []

            lane_mesh[lane.id].append(p_lane_inner)
            lane_mesh[lane.id].append(p_lane_outer)

        return lane_mesh

    def get_road_mesh(self, s_step=1.0, road_index=-1):

        mesh = {}
        roads = self.road if road_index == -1 else [road[road_index]]

        for road in roads:
            # section counter
            idx_sec = 0

            # create new mesh for section
            mesh[road.id] = {idx_sec: {}}

            # generate list of evaluation points along the reference line
            s_list = [np.float64(i * s_step)
                      for i in range(int(np.ceil(road.length / s_step)))]

            # include the end point
            s_list.append(road.length)

            while len(s_list):
                # determine maximum value of the current region
                s_max = road.lanes.laneSection[idx_sec+1].s if idx_sec < len(
                    road.lanes.laneSection) - 1 else road.length

                # check if the next point would exceed the boundaries, to deal with bound conditions
                if s_list[0] > s_max and s_max != road.length:
                    s_list.insert(0, s_max)
                    s_list.insert(0, s_max - 1.0e-6)

                # pop the next element
                s_int = min(s_list.pop(0), road.length)

                # Logic to advance to the next lane section, independently for left/right
                if idx_sec < len(road.lanes.laneSection) - 1 and s_int >= s_max:
                    idx_sec += 1
                    mesh[road.id][idx_sec] = {}

                # a section always contains a center tag
                sec_cen = road.lanes.laneSection[idx_sec]

                if (len(sec_cen.left) or not sec_cen.singleSide):
                    # this section is valid for the left side
                    sec_left = sec_cen
                    idx_sec_left = idx_sec

                if (len(sec_cen.right) or not sec_cen.singleSide):
                    # this section is valid for the right side
                    sec_right = sec_cen
                    idx_sec_right = idx_sec

                # get reference line position
                x, y, z, hdg, roll = road.get_reference_line(s_int)

                # get lane offset
                l_offset = road.lanes.get_lane_offset(s_int)

                # We need to determine the axis of the reference plane
                # Two vectors pointing into the lateral direction are required
                # 1. One is horizontal and thus only rotated with the heading angle
                #   (required for planar road lanes)
                # 2. Rotated with heading and considering road banking
                l_vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)

                th_vec = RotateZ(l_vec, hdg)
                t_vec = RotateZ(RotateX(l_vec, roll), hdg)

                # center reference point
                p_cen = np.array([x, y, z], copy=True)

                # get left side lane meshes
                lane_mesh = self.get_lane_bounds(s_int, sec_left, p_cen,
                                                 th_vec, t_vec, l_offset, left_side=True)

                # append data to existing mesh
                for k, v in lane_mesh.items():
                    if k not in mesh[road.id][idx_sec_left]:
                        mesh[road.id][idx_sec_left][k] = []
                    mesh[road.id][idx_sec_left][k] += v

                # get right side lane meshes
                lane_mesh = self.get_lane_bounds(s_int, sec_right, p_cen,
                                                 th_vec, t_vec, l_offset, left_side=False)

                # append data to existing mesh
                for k, v in lane_mesh.items():
                    if k not in mesh[road.id][idx_sec_right]:
                        mesh[road.id][idx_sec_right][k] = []
                    mesh[road.id][idx_sec_right][k] += v

        return mesh

    def get_road_mesh_new(self, s_step=1.0, road_index=-1):

        mesh = {}
        roads = self.road if road_index == -1 else [road[road_index]]

        """
        The idea is to evaluate all sections backwards, starting from the road length 
        and tracking backwards until the starting point of the section. 

        We then sample points along the reference line within this section.

        """

        sides = [1, -1]

        for road in roads:

            mesh[road.id] = {id: {}
                             for id in range(len(road.lanes.laneSection))}

            for side in sides:
                lane_sign = np.float64(side)

                s_max = road.length
                id_sec = len(road.lanes.laneSection)

                i_geo = len(road.planview.geometry) - 1
                i_ele = len(road.elevationProfile.elevation) - 1
                i_sup = len(road.lateralProfile.superelevation) - 1

                i_ofs = len(road.lanes.laneOffset) - 1

                # process left lanes
                while id_sec > 0:
                    while True:
                        # decrement section index
                        id_sec -= 1

                        lane_sec = road.lanes.laneSection[id_sec]

                        if side == 1:
                            lane_section_side = lane_sec.left[::-1]
                        elif side == -1:
                            lane_section_side = lane_sec.right
                        else:
                            lane_section_side = lane_sec.center

                        if len(lane_section_side) or id_sec == 0 or not lane_sec.singleSide:
                            break

                    s_min = lane_sec.s

                    # Drop short sections
                    if s_max - s_min < 1.0e-1:
                        continue

                    # compute evenly spaced distance intervals
                    # s_int = np.linspace(s_min, s_max - 1.0e-6, int(
                    #     np.ceil((s_max - s_min) / s_step)) + 1)
                    s_int = np.linspace(s_max - 1.0e-6, s_min, int(
                        np.ceil((s_max - s_min) / s_step)) + 1)

                    # TODO: Change this part to simply use a distance
                    # do take care of the nice and even spread
                    # s_delta = (s_max - s_min) / (np.floor((s_max - s_min) / s_step) + 1)
                    # do the following: s_int = s_max - s_delta * count until s_int >= s_min

                    for s in s_int:
                        # scan to start index of the reference line profiles
                        while i_geo > 0 and s < road.planview.geometry[i_geo].s:
                            i_geo -= 1
                        while i_ele > 0 and s < road.elevationProfile.elevation[i_ele].s:
                            i_ele -= 1
                        while i_sup > 0 and s < road.lateralProfile.superelevation[i_sup].s:
                            i_sup -= 1
                        while i_ofs > 0 and s < road.lanes.laneOffset[i_ofs]:
                            i_ofs -= 1

                        # get reference line position
                        # x, y, z, hdg, roll = road.get_reference_line(s)
                        x, y, hdg = road.planview.geometry[i_geo].get_xyt(s)
                        z = road.elevationProfile.elevation[i_ele].eval_s(
                            s) if i_ele >= 0 else np.float64(0)
                        roll = road.lateralProfile.superelevation[i_sup].eval_s(
                            s) if i_sup >= 0 else np.float64(0)

                        # get lane offset
                        # l_offset = road.lanes.get_lane_offset(s)
                        l_offset = road.lanes.laneOffset[i_ofs].eval_s(
                            s) if i_ofs >= 0 else np.float64(0)

                        l_vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                        th_vec = RotateZ(l_vec, hdg)
                        t_vec = RotateZ(RotateX(l_vec, roll), hdg)

                        u_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                        hh_vec = u_vec
                        h_vec = RotateZ(RotateX(u_vec, roll), hdg)

                        # calculate center reference point
                        p_lane = np.array([x, y, z], copy=True)
                        p_lane += l_offset * t_vec

                        for lane in lane_section_side:
                            # get lane width and height (inner, outer)
                            ds = s - s_min
                            lane_width = lane_sign * lane.get_lane_width(ds)
                            h_inner, h_outer = lane.get_lane_height(ds)

                            # effective lane lateral vector
                            t_lat = t_vec if not lane.level else th_vec
                            h_vrt = h_vec if not lane.level else hh_vec

                            # calculate bound coordinates
                            p_lane_inner = p_lane.copy() + h_inner * h_vrt
                            p_lane += lane_width * t_lat
                            p_lane_outer = p_lane.copy() + h_outer * h_vrt

                            # append to the mesh structure
                            if lane.id not in mesh[road.id][id_sec]:
                                mesh[road.id][id_sec][lane.id] = {
                                    'data': lane, 'in': [], 'out': []}

                            mesh[road.id][id_sec][lane.id]['in'].append(
                                p_lane_inner)
                            mesh[road.id][id_sec][lane.id]['out'].append(
                                p_lane_outer)

                    # store previous section length
                    s_max = s_min

        return mesh

    def get_roadmarker_mesh(self, s_step=1.0, road_index=-1):

        mesh = {}
        roads = self.road if road_index == -1 else [road[road_index]]

        """
        The idea is to evaluate all sections backwards, starting from the road length 
        and tracking backwards until the starting point of the section. 

        We then sample points along the reference line within this section.

        """

        sides = [1, 0, -1]

        for road in roads:

            mesh[road.id] = {id: {}
                             for id in range(len(road.lanes.laneSection))}

            for side in sides:
                lane_sign = np.float64(side)

                s_max = road.length

                lane_sections = road.lanes.laneSection.copy()

                # process left lanes
                while len(lane_sections):
                    while True:
                        # decrement section index
                        lane_section = lane_sections.pop()

                        if side == 1:
                            lane_section_side = lane_section.left[::-1]
                        elif side == -1:
                            lane_section_side = lane_section.right
                        else:
                            lane_section_side = lane_section.center

                        if len(lane_section_side) or not len(lane_sections) or not lane_section.singleSide:
                            break

                    id_sec = len(lane_sections)

                    s_min = lane_section.s

                    # drop short sections
                    if s_max - s_min < 1.0e-1:
                        continue

                    for lane_cnt, lane in enumerate(lane_section_side):
                        # sOffset is relative to the start of the section
                        s_offset_max = s_max - s_min

                        road_marks = lane.roadmark.copy()

                        if lane.id not in mesh[road.id][id_sec]:
                            mesh[road.id][id_sec][lane.id] = {}

                        while len(road_marks):
                            marker = road_marks.pop()
                            marker_id = len(road_marks)

                            s_offset_min = marker.sOffset

                            # compute evenly spaced distance interval
                            s_int = np.linspace(s_offset_min, s_offset_max, int(
                                np.ceil((s_offset_max - s_offset_min) / s_step)) + 1)

                            # store previous offset
                            s_offset_max = s_offset_min

                            for s_ofs in s_int:
                                s = s_ofs + lane_section.s

                                # get reference line position
                                x, y, z, hdg, roll = road.get_reference_line(s)

                                # get lane offset
                                l_offset = road.lanes.get_lane_offset(s)

                                l_vec = np.array(
                                    [0.0, 1.0, 0.0], dtype=np.float64)
                                th_vec = RotateZ(l_vec, hdg)
                                t_vec = RotateZ(RotateX(l_vec, roll), hdg)

                                u_vec = np.array(
                                    [0.0, 0.0, 1.0], dtype=np.float64)
                                hh_vec = u_vec
                                h_vec = RotateZ(RotateX(u_vec, roll), hdg)

                                # calculate center reference point
                                p_ref = np.array([x, y, z], copy=True)

                                # get accumulated lane widths
                                lane_bound = l_offset + lane_sign * \
                                    np.sum([l.get_lane_width(
                                        s_ofs) for l in lane_section_side[0:lane_cnt+1]])

                                _, h_outer = lane.get_lane_height(s_ofs)

                                # get the marker width
                                marker_width = marker.width
                                marker_height = marker.height

                                # effective lane lateral vector
                                t_lat = t_vec if not lane.level else th_vec
                                h_vrt = h_vec if not lane.level else hh_vec

                                # calculate lane marker coordinates
                                p_lane_outer = p_ref + lane_bound * t_lat + \
                                    (h_outer + marker_height) * h_vrt

                                p_marker_inner = p_lane_outer.copy() - marker_width / 2 * t_lat
                                p_marker_outer = p_lane_outer.copy() + marker_width / 2 * t_lat

                                # append to the mesh structure
                                if marker_id not in mesh[road.id][id_sec][lane.id]:
                                    mesh[road.id][id_sec][lane.id][marker_id] = {
                                        'data': marker, 'in': [], 'out': []}

                                mesh[road.id][id_sec][lane.id][marker_id]['in'].append(
                                    p_marker_inner)
                                mesh[road.id][id_sec][lane.id][marker_id]['out'].append(
                                    p_marker_outer)

                    # store previous section length
                    s_max = s_min

        return mesh

    @ staticmethod
    def parse(file):
        od = OpenDrive()
        xodr = ET.parse(open(file)).getroot()
        for node in xodr:
            tag = node.tag
            if tag == "road":
                rd = Road.parse(node)
                od.road.append(rd)
        return od

    @ staticmethod
    def from_string(string):
        od = OpenDrive()
        xodr = ET.fromstring(string)
        for node in xodr:
            tag = node.tag
            if tag == "road":
                rd = Road.parse(node)
                od.road.append(rd)
        return od


if __name__ == "__main__":

    path = os.path.dirname(__file__)
    file = os.path.join(path, "./data", "Roundabout8Course.xodr")
    # file = os.path.join(path, "./data", "Crossing8Course.xodr")
    # file = os.path.join(path, "./data", "Town4.xodr")
    # file = os.path.join(path, "./data", "CrossingComplex8Course.xodr")

    od = OpenDrive.parse(file)
    od = od

    from datetime import datetime

    t_start = datetime.now()

    # rm = od.get_road_mesh()

    rm = od.get_road_mesh_new()
    t_ela_1 = (datetime.now() - t_start).microseconds / 1000

    mk = od.get_roadmarker_mesh()

    t_ela = (datetime.now() - t_start).microseconds / 1000

    print(t_ela)
    print(t_ela_1)

    # import json
    # print(json.dumps(rm, indent=3))
