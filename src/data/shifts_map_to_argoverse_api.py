from cmath import nan
from typing import List, Union
import numpy as np

from argoverse.utils.manhattan_search import (
    find_all_polygon_bboxes_overlapping_query_bbox,
)
from ysdc_dataset_api.utils.map import get_section_to_state, get_lane_availability


# Any numeric type
Number = Union[int, float]


def bbox_from_polygon(polygon):

    return [min(polygon[:, 0]), min(polygon[:, 1]), max(polygon[0]), max(polygon[1])]


def bbox_table_from_polygons(polygons):

    bbox_table = []

    for p in polygons:
        bbox_table.append(bbox_from_polygon(p))

    return np.array(bbox_table)


class ShiftsMap:
    """
    This class provides the interface to load a ShiftsMap in Argoverse format.

    """

    def __init__(self, path_graph, traffic_lights) -> None:
        """
        Initialize the Shifts Map.

        """

        self.path_graph = path_graph
        self.traffic_lights = traffic_lights

        # Get state of traffic lights at last observed time
        self.traffic_light_states = get_section_to_state(traffic_lights[24])

        # lane centerlines
        self.lanes_centerline, self.lanes_max_vel, self.lanes_give_way, self.lanes_availability = self.get_lanes()

        # lane bbox
        self.lanes_bbox = bbox_table_from_polygons(self.lanes_centerline)

        # road polygons
        # drivable area is not included in the Shifts map
        self.road_polygons = self.get_road_polygons()

        # crosswalk polygons
        self.crosswalk_polygons = self.get_crosswalk_polygons()


    def get_lanes(self):
        """
        Get list of lanes.
        
        Return:
            List arrays, each is an array of [Kx3], <x, y, z>.
        """

        lanes_centerline = []
        lanes_max_vel = []
        lanes_give_way = []
        lanes_availability = []


        for lane in self.path_graph.lanes:
            
            # centerline
            curr_lane_centerline = []  # [Kx3], <x, y, z>

            for p in lane.centers:
                curr_lane_centerline.append([p.x, p.y, nan])
            
            lanes_centerline.append(np.array(curr_lane_centerline))

            # max_vel
            lanes_max_vel.append(lane.max_velocity)

            # gives way
            lanes_give_way.append(lane.gives_way_to_some_lane)

            # availability (traffic light)
            lanes_availability.append(get_lane_availability(lane, self.traffic_light_states))

        return lanes_centerline, lanes_max_vel, lanes_give_way, lanes_availability


    def get_road_polygons(self):
        """
        Get list of road polygons.
        
        Returns:
            List of polygons, each is an array of [Kx3], <x, y, z>.
        """

        road_polygons = []

        for road_polygon in self.path_graph.road_polygons:
            
            rp = []

            for p in road_polygon.geometry.points:
                rp.append([p.x, p.y, nan])

            road_polygons.append(np.array(rp))

        return road_polygons


    def get_crosswalk_polygons(self):
        """
        Get list of crosswalk polygons.
        
        Returns:
            List of polygons, each is an array of [Kx3], <x, y, z>.
        """

        crosswalk_polygons = []

        for crosswalk_polygon in self.path_graph.crosswalks:
            
            cwp = []

            for p in crosswalk_polygon.geometry.points:
                cwp.append([p.x, p.y, nan])

            crosswalk_polygons.append(np.array(cwp))

        return crosswalk_polygons


    def get_lane_centerline(self, lane_id):

        return self.lanes_centerline[lane_id]
    

    def get_lane_max_vel(self, lane_id):

        return self.lanes_max_vel[lane_id]
    

    def get_lane_give_way(self, lane_id):

        return self.lanes_give_way[lane_id]


    def get_lane_availability(self, lane_id):

        availability = self.lanes_availability[lane_id]
        if availability is None:
            availability = 0.0
        
        # Availability goes from -2 to 2 (this is defined in the shifts api)
        # We just offset it to go from 0 to 5 (in HiVT an embedding is learned)
        availability += 2.0

        return availability


    def get_lane_ids_in_xy_bbox(
        self,
        query_x: float,
        query_y: float,
        query_search_range_manhattan: float = 5.0,
    ) -> List[int]:
        """
        Taken from: https://github.com/argoai/argoverse-api/blob/master/argoverse/map_representation/map_api.py#:~:text=def%20get_lane_ids_in_xy_bbox(
        
        Prune away all lane segments based on Manhattan distance. We vectorize this instead
        of using a for-loop. Get all lane IDs within a bounding box in the xy plane.
        This is a approximation of a bubble search for point-to-polygon distance.

        The bounding boxes of small point clouds (lane centerline waypoints) are precomputed in the map.
        We then can perform an efficient search based on manhattan distance search radius from a
        given 2D query point.

        We pre-assign lane segment IDs to indices inside a big lookup array, with precomputed
        hallucinated lane polygon extents.

        Args:
            query_x: representing x coordinate of xy query location
            query_y: representing y coordinate of xy query location
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
            query_search_range_manhattan: search radius along axes

        Returns:
            lane_ids: lane segment IDs that live within a bubble
        """
        query_min_x = query_x - query_search_range_manhattan
        query_max_x = query_x + query_search_range_manhattan
        query_min_y = query_y - query_search_range_manhattan
        query_max_y = query_y + query_search_range_manhattan

        neighborhood_lane_ids = find_all_polygon_bboxes_overlapping_query_bbox(
            self.lanes_bbox,
            np.array([query_min_x, query_min_y, query_max_x, query_max_y]),
        )

        if len(neighborhood_lane_ids) == 0:
            return []

        return neighborhood_lane_ids
