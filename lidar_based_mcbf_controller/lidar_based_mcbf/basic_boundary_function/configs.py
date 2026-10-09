import os
import pathlib
from abc import ABC, abstractmethod
from typing import Any, Union
import yaml
import numpy as np


from ament_index_python.packages import get_package_share_directory

def load_map_config(env_name: str, config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(get_package_share_directory('lidar_based_mcbf'), 'config', 'env_maps.yaml')
    with open(config_path, 'r') as f:
        env_maps = yaml.safe_load(f)
    if env_name not in env_maps:
        raise ValueError(f"Unknown environment name: {env_name}. "
                         f"Add it to {config_path}; available: {list(env_maps)}")
    return env_maps[env_name]

class Configurator:
    FIRST_LOAD = False

    def __init__(self, config_fp: str, config_type:str='yaml', with_partition=False) -> None:
        if Configurator.FIRST_LOAD:
            print(f'{self.__class__.__name__} Loading configuration from "{config_fp}".')
            Configurator.FIRST_LOAD = False

        if config_type == 'yaml':
            if with_partition:
                yaml_load = self.from_yaml_all(config_fp)
            else:
                yaml_load = self.from_yaml(config_fp)
            for key in yaml_load:
                setattr(self, key, yaml_load[key])

    def set_extra_attr(self, key, value):
        setattr(self, key, value)

    @staticmethod
    def from_yaml(load_path) -> Union[dict, Any]:
        with open(load_path, 'r') as stream:
            try:
                parsed_yaml = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return parsed_yaml

    @staticmethod
    def from_yaml_all(load_path) -> Union[dict, Any]:
        parsed_yaml = {}
        with open(load_path, 'r') as stream:
            try:
                for data in yaml.load_all(stream, Loader=yaml.FullLoader):
                    parsed_yaml.update(data)
            except yaml.YAMLError as exc:
                print(exc)
        return parsed_yaml


class _Configuration(ABC):
    def __init__(self, config: Configurator, manual_load=False) -> None:
        self._config = config
        if not manual_load:
            self._load_config()

    @abstractmethod
    def _load_config(self):
        pass

    def manual_load(self):
        self._load_config()

    @classmethod
    def from_yaml(cls, yaml_fp: str, with_partition=False):
        config = Configurator(yaml_fp, with_partition=with_partition)
        return cls(config)


class WarehouseSimConfiguration(_Configuration):
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.scene = config.scene
        self.map_dir = config.map_dir
        self.map_file = config.map_file
        self.graph_file = config.graph_file

        self.sim_width = config.sim_width
        self.sim_height = config.sim_height

        self.scale2nn = config.scale2nn
        self.scale2real = config.scale2real

        self.image_axis = config.image_axis
        self.corner_coords = config.corner_coords

class CircularRobotSpecification(_Configuration):
    def __init__(self, config: Configurator):
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts

        self.rob_name = config.rob_name
        self.vehicle_width = config.vehicle_width
        self.vehicle_margin = config.vehicle_margin
        self.social_margin = config.social_margin
        self.lin_vel_min = config.lin_vel_min
        self.lin_vel_max = config.lin_vel_max
        self.lin_acc_min = config.lin_acc_min
        self.lin_acc_max = config.lin_acc_max
        self.ang_vel_max = config.ang_vel_max
        self.ang_acc_max = config.ang_acc_max
        self.ang_vel_onm_max = config.ang_vel_onm_max
        self.lidar_range = config.lidar_range

class PedestrianSpecification(_Configuration):
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts

        self.human_width = config.human_width
        self.human_vel_max = config.human_vel_max
        self.human_stagger = config.human_stagger

class MpcConfiguration(_Configuration):
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts

        self.N_hor = config.N_hor
        self.action_steps = config.action_steps

        self.ns = config.ns
        self.nu = config.nu
        self.nq = config.nq
        self.Nother = config.Nother
        self.nstcobs = config.nstcobs
        self.Nstcobs = config.Nstcobs
        self.ndynobs = config.ndynobs
        self.Ndynobs = config.Ndynobs

        self.solver_type = config.solver_type

        self.max_solver_time = config.max_solver_time
        self.build_directory = config.build_directory
        self.build_type = config.build_type
        self.bad_exit_codes = config.bad_exit_codes
        self.optimizer_name = config.optimizer_name

        self.lin_vel_penalty = config.lin_vel_penalty
        self.lin_acc_penalty = config.lin_acc_penalty
        self.ang_vel_penalty = config.ang_vel_penalty
        self.ang_acc_penalty = config.ang_acc_penalty
        self.qrpd = config.qrpd
        self.qpos = config.qpos
        self.qvel = config.qvel
        self.qtheta = config.qtheta
        self.qstcobs = config.qstcobs
        self.qdynobs = config.qdynobs
        self.qpN = config.qpN
        self.qthetaN = config.qthetaN

class GPDFConfiguration(_Configuration):
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.d_offset_0 = config.D_OFFSET_0
        self.L = config.L
        self.rho = config.Rho

class EnvContrConfiguration(_Configuration):
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.dynamics = config.dynamics
        self.env_name = config.env_name
        self.init_state = config.init_state
        self.target = np.array(config.target)
        self.radius = config.radius
        self.MCBF = config.MCBF
        self.nominal_speed = config.nominal_speed

        contr_path = os.path.join(get_package_share_directory("lidar_based_mcbf"), "config", "autotune_2D.yaml")
        contr = ContrConfiguration.from_yaml(contr_path)
        self.w = contr.w
        self.iter = contr.iter
        self.iter_g = contr.iter_g
        self.iter_c = contr.iter_c
        self.hold_time = contr.hold_time
        self.base_margin = contr.base_margin
        self.target_range = contr.target_range
        self.om_range = contr.om_range
        self.ve = contr.ve
        self.beta_coef = contr.beta_coef
        self.om_expand_threshold = contr.om_expand_threshold
        self.dir_num = contr.dir_num
        self.custom_iter1 = contr.custom_iter1
        self.custom_iter2 = contr.custom_iter2

class ContrConfiguration(_Configuration):
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.w = config.w
        self.iter = config.iter
        self.iter_g = config.iter_g
        self.iter_c = config.iter_c
        self.hold_time = config.hold_time
        self.base_margin = config.base_margin
        self.target_range = config.target_range
        self.om_range = config.om_range
        self.ve = config.ve
        self.beta_coef = config.beta_coef
        self.om_expand_threshold = config.om_expand_threshold
        self.dir_num = config.dir_num
        self.custom_iter1 = config.custom_iter1
        self.custom_iter2 = config.custom_iter2


class TebConfiguration(_Configuration):
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts

        self.N_hor = config.N_hor
        self.N_obs = config.N_obs

class RPPConfiguration(_Configuration):
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts

        self.lt = config.lt
        self.Tk = config.Tk
        self.r_min = config.r_min
        self.d_prox = config.d_prox
        self.vt = config.vt

        self.N_hor = config.N_hor

class SGANConfig(_Configuration):
    def __init__(self, config: Configurator):
        super().__init__(config, manual_load=True)
        self._extra_init()

    def _extra_init(self):
        for key in self._config.args:
            self._config.set_extra_attr(key, self._config.args[key])
        self.manual_load()

    def _load_config(self):
        config = self._config

        self.dataset_name = config.dataset_name
        self.skip = config.skip
        self.delim = config.delim
        self.batch_size = config.batch_size
        self.loader_num_workers = config.loader_num_workers

        self.obs_len = config.obs_len
        self.pred_len = config.pred_len
        self.embedding_dim = config.embedding_dim
        self.encoder_h_dim = config.encoder_h_dim_g
        self.decoder_h_dim = config.decoder_h_dim_g
        self.mlp_dim = config.mlp_dim
        self.num_layers = config.num_layers
        self.noise_dim = tuple(config.noise_dim)
        self.noise_type = config.noise_type
        self.noise_mix_type = config.noise_mix_type
        self.pooling_type = config.pooling_type
        self.pool_every_timestep = config.pool_every_timestep
        self.dropout = config.dropout
        self.bottleneck_dim = config.bottleneck_dim
        self.neighborhood_size = config.neighborhood_size
        self.grid_size = config.grid_size
        self.batch_norm = config.batch_norm


