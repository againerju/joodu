import json
import os
from itertools import permutations, product
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from sdc.filters import filter_ood_development_data, filter_ood_evaluation_data
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from ysdc_dataset_api.utils import get_file_paths, read_scene_from_file

from src.data.shifts_map_to_argoverse_api import ShiftsMap
from utils import TemporalData


class ShiftsDataset(Dataset):

    def __init__(self,
                 root: str = None,
                 split: str = None,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50
                 ) -> None:

        self.root = root
        self._split = split
        self._local_radius = local_radius
        self._directory = self.get_directory()
        
        self._raw_file_paths = get_file_paths(self.raw_dir)
        self._raw_file_names = [os.path.join(*f.split(os.sep)[-2:]) for f in self._raw_file_paths]
        self._processed_file_names = [os.path.splitext(f)[0] + ".pt" for f in self._raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]

        # read tags file
        self._scene_tags_fpath = os.path.join(self.raw_dir, "tags.txt")
        self._scene_tags = []
        with open(self._scene_tags_fpath, 'r') as f:
            for i, line in enumerate(f):
                self._scene_tags.append(json.loads(line.strip()))

        super(ShiftsDataset, self).__init__(root, transform=transform)


    def get_directory(self) -> str:

        self.directory_map = dict()
        self.directory_map["train"] = "canonical-trn-dev-data/train"
        self.directory_map["dev"] = "canonical-trn-dev-data/development"
        self.directory_map["eval"] = "canonical-eval-data/evaluation"

        return self.directory_map[self._split]


    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory)


    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed')


    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names


    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names


    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths


    def create_process_args(self):

        args = []

        for i, raw_path in enumerate(self.raw_paths):

            args.append([raw_path, self._split, self._scene_tags[i], self._local_radius, self.processed_paths[i]])

        return args


    def process(self) -> None:
        args = self.create_process_args()

        for arg in tqdm(args):
            process_shifts(arg)


    def len(self) -> int:
        return len(self._raw_file_names)


    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])


def init_trajectory_dict():

    return {
            "TIMESTAMP": [],
            "TRACK_ID": [],
            "OBJECT_TYPE": [],
            "VEHICLE": [],
            "X": [],
            "Y": [],
            "VX": [],
            "VY": [],
            "AX": [],
            "AY": [],
            "YAW": []
          }


def append_trajectory_dict(trajectory_dict, timestamp, track_id, object_type, vehicle, x, y, vx, vy, ax, ay, yaw):

    trajectory_dict["TIMESTAMP"].append(timestamp)
    trajectory_dict["TRACK_ID"].append(track_id)
    trajectory_dict["OBJECT_TYPE"].append(object_type)
    trajectory_dict["VEHICLE"].append(vehicle)
    trajectory_dict["X"].append(x)
    trajectory_dict["Y"].append(y)
    trajectory_dict["VX"].append(vx)
    trajectory_dict["VY"].append(vy)
    trajectory_dict["AX"].append(ax)
    trajectory_dict["AY"].append(ay)
    trajectory_dict["YAW"].append(yaw)

    return trajectory_dict


def encode_shifts_object_type(sobject_type):

    return {"ego": "AV", "vehicle": "VEHICLE", "pedestrian": "PEDESTRIAN"}[sobject_type]


def extract_ego_vehicle_trajectory(track, t_init=0, ctrack_id=None, sobject_type="ego"):
    """ Return the trajectory given the ego vehicle track.
    Arguments:
    track -- protobuf track.

    """

    trajectory_dict = init_trajectory_dict()

    for t, agent in enumerate(track):
        timestamp = t_init + t
        track_id = ctrack_id
        object_type = encode_shifts_object_type(sobject_type)
        vehicle = 1
        x, y, z, xdim, ydim, zdim, vx, vy, ax, ay, yaw = get_agent_state(agent, object_type="ego")

        trajectory_dict = append_trajectory_dict(trajectory_dict, timestamp, track_id, object_type, vehicle, x, y, vx, vy, ax, ay, yaw)

    df = pd.DataFrame.from_dict(trajectory_dict)

    return df


def get_agent_state(agent, object_type):

    x = agent.position.x
    y = agent.position.y
    z = agent.position.z
    xdim = agent.dimensions.x
    ydim = agent.dimensions.y
    zdim = agent.dimensions.z
    vx = agent.linear_velocity.x
    vy = agent.linear_velocity.y

    if object_type == "vehicle":
        ax = agent.linear_acceleration.x
        ay = agent.linear_acceleration.y
        yaw = agent.yaw
    elif object_type == "ego":
        ax = 0
        ay = 0
        yaw = agent.yaw
    elif object_type == "pedestrian":
        ax, ay, yaw = 0, 0, 0
    else:
        raise ValueError("Invalid object type " + object_type)

    return x, y, z, xdim, ydim, zdim, vx, vy, ax, ay, yaw


def extract_trajectories_from_pb_tracks(pb_tracks, t_init=0, sobject_type="vehicle"):

    agent_trajectories = init_trajectory_dict()

    for t, timestep in enumerate(pb_tracks):
        for agent in timestep.tracks:

            timestamp = t_init + t
            track_id = agent.track_id
            object_type = encode_shifts_object_type(sobject_type)

            x, y, z, xdim, ydim, zdim, vx, vy, ax, ay, yaw = get_agent_state(agent, object_type=sobject_type)

            vehicle = 1
            if sobject_type == "pedestrian":
                vehicle = 0

            agent_trajectories = append_trajectory_dict(agent_trajectories, timestamp, track_id, object_type, vehicle, x, y, vx, vy, ax, ay, yaw)

    # convert to dataframe
    df = pd.DataFrame.from_dict(agent_trajectories)

    return df


def get_unique_track_id(unique_track_ids):

    new_track_id = 0

    while new_track_id in unique_track_ids:
        new_track_id += 1

    return new_track_id


def set_object_type(df):

    df.loc[df["OBJECT_TYPE"] != "AV", "OBJECT_TYPE"] = "OTHERS"

    return df


def set_target_agents(df, prediction_requests):

    df.loc[(df["TRACK_ID"].isin(prediction_requests)) & (df["VEHICLE"] == 1), "OBJECT_TYPE"] = "AGENT"  # only vehicles are predicted in shifts

    return df


def get_trajectory_tags():

    tag_map = dict()
    tag_map[0] = "MoveLeft"
    tag_map[1] = "MoveRight"
    tag_map[2] = "MoveForward"
    tag_map[3] = "MoveBack"
    tag_map[4] = "Acceleration"
    tag_map[5] = "Deceleration"
    tag_map[6] = "Uniform"
    tag_map[7] = "Stopping"
    tag_map[8] = "Starting"
    tag_map[9] = "Stationary"

    return tag_map


def get_one_hot_trajectory_tags_mat(prediction_requests):

    one_hot_tags_mat = torch.zeros((len(prediction_requests), len(get_trajectory_tags())), dtype=torch.bool)

    for i, pr in enumerate(prediction_requests):
        one_hot_tags_mat[i, pr.trajectory_tags] = True

    return one_hot_tags_mat


def process_shifts(args: list) -> None:

    # args
    raw_path = args[0]
    split = args[1]
    scene_tags = args[2]
    radius = args[3]
    processed_path = args[4]

    # read protobuf
    pb = read_scene_from_file(raw_path)

    # map
    sm = ShiftsMap(pb.path_graph, pb.traffic_lights)

    # store all tracks in dataframe

    # vehicle tracks
    df_past_vehicle_tracks = extract_trajectories_from_pb_tracks(pb.past_vehicle_tracks, t_init=0, sobject_type="vehicle")  # [N(t)*T(N), 11]
    df_future_vehicle_tracks = extract_trajectories_from_pb_tracks(pb.future_vehicle_tracks, t_init=25, sobject_type="vehicle")  # [N(t)*T(N), 11]

    # pedestrian tracks
    df_past_pedestrian_tracks = extract_trajectories_from_pb_tracks(pb.past_pedestrian_tracks, t_init=0, sobject_type="pedestrian")  # [N(t)*T(N), 11]
    df_future_pedestrian_tracks = extract_trajectories_from_pb_tracks(pb.future_pedestrian_tracks, t_init=25, sobject_type="pedestrian")  # [N(t)*T(N), 11]

    # combine dataframes
    df = pd.concat([df_past_vehicle_tracks, df_future_vehicle_tracks, df_past_pedestrian_tracks, df_future_pedestrian_tracks], ignore_index=True)

    # determine ego track id
    all_track_ids = df["TRACK_ID"].unique()
    ego_track_id = get_unique_track_id(all_track_ids)

    # ego track, T_o = observation time, T_p = prediction time
    df_past_ego_trajectory = extract_ego_vehicle_trajectory(pb.past_ego_track, t_init=0, ctrack_id=ego_track_id, sobject_type="ego")  # [T_o, 11]
    df_future_ego_trajectory = extract_ego_vehicle_trajectory(pb.future_ego_track, t_init=25, ctrack_id=ego_track_id, sobject_type="ego")  # [T_p, 11]

    # combine dataframes
    df = pd.concat([df, df_past_ego_trajectory, df_future_ego_trajectory], ignore_index=True)

    # add city to dataframe
    df["CITY_NAME"] = [scene_tags['track'].upper()]*len(df)

    # set agent type in argoverse format {AV, OTHERS}
    df = set_object_type(df)

    # set multiple target agents {AGENT} according to prediction requests
    prediction_requests = [r.track_id for r in pb.prediction_requests]
    df = set_target_agents(df, prediction_requests)

    # target agent maneuvers
    tags = get_one_hot_trajectory_tags_mat(pb.prediction_requests)
       
    # add historical time stamps
    n_past_ts = 25
    n_futr_ts = 25
    n_total_ts = n_past_ts + n_futr_ts
    
    # filter out actors that are unseen during the historical time steps
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    assert len(timestamps) == (n_total_ts)
    historical_timestamps = timestamps[: n_past_ts]  # in shifts 25 historical time steps = 5s @ 5 Hz
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
    historical_track_id_vehicle_df = historical_df[["TRACK_ID", "VEHICLE"]].drop_duplicates()
    num_nodes = historical_track_id_vehicle_df.shape[0]  # track-ids are not unique, might be the same for vehicle and pedestrian
    select_valid_actors = [(item.tolist() in historical_track_id_vehicle_df.values.tolist()) for item in df[["TRACK_ID", "VEHICLE"]].values]
    df = df[select_valid_actors]
    actor_identifications = historical_track_id_vehicle_df.values.tolist()
    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc
    av_index = actor_identifications.index([av_df[0]['TRACK_ID'], av_df[0]['VEHICLE']])
    agent_index = [i for i in range(len(actor_identifications)) if actor_identifications[i][0] in prediction_requests and actor_identifications[i][1] == 1]
    city = df['CITY_NAME'].values[0]

    # make the scene centered at AV
    origin = torch.tensor([av_df[n_past_ts-1]['X'], av_df[n_past_ts-1]['Y']], dtype=torch.float)
    av_heading_vector = origin - torch.tensor([av_df[n_past_ts-2]['X'], av_df[n_past_ts-2]['Y']], dtype=torch.float)
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                            [torch.sin(theta), torch.cos(theta)]])

    # initialization
    x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
    v = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
    a = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
    yaw = torch.zeros(num_nodes, 50, 1, dtype=torch.float)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 25, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
    valid = torch.zeros(num_nodes, dtype=torch.bool)
    is_vehicle = torch.zeros(num_nodes, dtype=torch.bool)

    for group_id, actor_df in df.groupby(['TRACK_ID', 'VEHICLE']):
        node_idx = actor_identifications.index(list(group_id))
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]
        padding_mask[node_idx, node_steps] = False  # no padding only when agent is present
        if padding_mask[node_idx, 24]:  # make no predictions for actors that are unseen at the current time step
            padding_mask[node_idx, 25:] = True
        # position (m)
        xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        # velocity  (in m/s)
        vxy = torch.from_numpy(np.stack([actor_df['VX'].values, actor_df['VY'].values], axis=-1)).float()
        v[node_idx, node_steps] = torch.matmul(vxy, rotate_mat)
        # acceleration (in m/s^2)
        axy = torch.from_numpy(np.stack([actor_df['AX'].values, actor_df['AY'].values], axis=-1)).float()
        a[node_idx, node_steps] = torch.matmul(axy, rotate_mat)
        # yaw
        yaw[node_idx, node_steps] = torch.from_numpy(actor_df["YAW"].values)[:, None].float()
        # historical time steps
        node_historical_steps = list(filter(lambda node_step: node_step < 25, node_steps))
        # is vehicle
        is_vehicle[node_idx] = torch.tensor(actor_df['VEHICLE'].iloc[0], dtype=torch.bool)
        # padding future, iff agent is not available in history at all
        if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        elif len(node_historical_steps) == 1:  # only available at one single timestep, use velocity for the heading vector
            heading_vector = v[node_idx, node_historical_steps[-1]]
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:  # make no predictions for the actor if the number of valid time steps is less than 1
            padding_mask[node_idx, 25:] = True
        # valid, if OBEJECT_TYPE==AGENT and available in all future time steps
        valid[node_idx] = (actor_df['OBJECT_TYPE'].iloc[0] == 'AGENT' and (~padding_mask[node_idx, 25:]).all())

    # bos_mask is True if time step t is valid and time step t-1 is invalid, denoting the start of a sequence
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1: 25] = padding_mask[:, : 24] & ~padding_mask[:, 1: 25]

    positions = x.clone()
    # if actor is not available in current time step set all future values to zero
    # if actor is available in current time step set all time steps to zero where the agent is not present
    # otherwise set to vectorized position 
    x[:, 25:] = torch.where((padding_mask[:, 24].unsqueeze(-1) | padding_mask[:, 25:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 25, 2),
                            x[:, 25:] - x[:, 24].unsqueeze(-2))
    # same for history
    x[:, 1: 25] = torch.where((padding_mask[:, : 24] | padding_mask[:, 1: 25]).unsqueeze(-1),
                            torch.zeros(num_nodes, 24, 2),
                            x[:, 1: 25] - x[:, : 24])
    # starting at zero
    x[:, 0] = torch.zeros(num_nodes, 2)

    # get lane features at the current time step
    # lane_vectors, max_vels, give_ways, availability, lane_actor_index, lane_actor_vectors
    df_24 = df[df['TIMESTAMP'] == timestamps[24]]
    node_inds_24 = [actor_identifications.index(list(actor_identification)) for actor_identification in df_24[['TRACK_ID', 'VEHICLE']].values.tolist()]
    node_positions_24 = torch.from_numpy(np.stack([df_24['X'].values, df_24['Y'].values], axis=-1)).float()

    (lane_vectors, max_vels, give_ways, availability, lane_actor_index,
    lane_actor_vectors) = get_lane_features(sm, node_inds_24, node_positions_24, origin, rotate_mat, city, radius)

    # target
    N = x.shape[0]
    y = x[:, 25:]
    seq_id = os.path.splitext(os.path.basename(raw_path))[0]

    # id or ood
    if "eval" in split:
        ood = filter_ood_evaluation_data(scene_tags_dict=scene_tags)
    elif "dev" in split:
        ood = filter_ood_development_data(scene_tags_dict=scene_tags)
    elif "train" in split:
        ood = False
    else:
        raise ValueError

    ood = np.repeat(ood, N)

    # store data
    output_data = {
                    'x': x[:, : 25],  # [N, 25, 2]
                    'positions': positions,  # [N, 50, 2]
                    'velocities': v,  # [N, 50, 2]
                    'accelerations': a, # [N, 50, 2]
                    'yaw': yaw,  # [N, 50, 1]
                    'edge_index': edge_index,  # [2, N x (N - 1)]
                    'y': y,  # [N, 25, 2]
                    'num_nodes': num_nodes, # = N
                    'padding_mask': padding_mask,  # [N, 50]
                    'bos_mask': bos_mask,  # [N, 25]
                    'rotate_angles': rotate_angles,  # [N]
                    'lane_vectors': lane_vectors,  # [L, 2]
                    'max_vels': max_vels,  # [L]
                    'give_ways': give_ways,  # [L]
                    'availability': availability,  # [L]
                    'lane_actor_index': lane_actor_index,  # [2, num_actor_lane_edges]
                    'lane_actor_vectors': lane_actor_vectors,  # [num_actor_lane_edges, 2]
                    'seq_id': seq_id,  # str
                    'track_id': torch.tensor(prediction_requests),  # [M], for M target agents   
                    'av_index': av_index,  # int
                    'is_vehicle': is_vehicle, # bool [N], True if agent is a vehicle (and not a pedestrian) 
                    'agent_index': torch.tensor(agent_index),  # int [M], for M target agents
                    'valid': valid[agent_index],  # bool [M], True if target agent is present in entire future
                    'tags': tags,  # bool [Mx#tags], one if agent is performing the maneuver, zero else
                    'city': city,  # str
                    'origin': origin.unsqueeze(0),  # [1x2]
                    'theta': theta,  # []
                    'ood': torch.tensor(ood),  # [N]
                    'N': N, # int
                    'raw_path': raw_path,  # path to raw protobuf file
                }

    # save
    export_processed(output_data, processed_path)


def export_processed(kwargs, processed_path):
    data = TemporalData(**kwargs)

    os.makedirs(os.path.split(processed_path)[0], exist_ok=True)

    output_path = os.path.join(processed_path)
    torch.save(data, output_path)


def get_lane_features(sm: ShiftsMap,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      city: str,
                      radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    lane_positions, lane_vectors, max_vels, give_ways, availability = [], [], [], [], []
    lane_ids = set()
    for node_position in node_positions:
        # get lanes within radius around node center
        lane_ids.update(sm.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], radius))
    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()
    for lane_id in lane_ids:
        lane_centerline = torch.from_numpy(sm.get_lane_centerline(lane_id)[:, :2]).float()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)

        # Add other lane properties
        lane_max_vel = sm.get_lane_max_vel(lane_id)
        lane_give_way = sm.get_lane_give_way(lane_id)
        lane_availability = sm.get_lane_availability(lane_id)

        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1

        max_vels.append(lane_max_vel * torch.ones(count, dtype=torch.uint8))
        give_ways.append(lane_give_way * torch.ones(count, dtype=torch.uint8))
        availability.append(lane_availability * torch.ones(count, dtype=torch.uint8))

    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    max_vels = torch.cat(max_vels, dim=0)
    give_ways = torch.cat(give_ways, dim=0)
    availability = torch.cat(availability, dim=0)

    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]

    return lane_vectors, max_vels, give_ways, availability, lane_actor_index, lane_actor_vectors


def rotation_mat_from_angle(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
