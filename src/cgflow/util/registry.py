from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from omegaconf import OmegaConf

C = TypeVar("C", bound=type[Any])
Dataclass = type[Any]


class Registry:
    """Registry to store class with config"""

    __obj_dict__: dict[str, "Registry"] = {}

    def __init__(self, name: str):
        assert name not in self.__obj_dict__, f'Registry name "{name}" already exists!'
        self.name: str = name
        self.__obj_dict__[name] = self  # add registry
        self._module_dict: dict[str, Any] = dict()

    @classmethod
    def get_register(cls, name: str) -> "Registry":
        assert name in cls.__obj_dict__, f"no registry with name '{name}' found"
        return cls.__obj_dict__[name]

    def __len__(self):
        return len(self._module_dict)

    def __getitem__(self, name: str) -> Any:
        module = self._module_dict.get(name, None)
        if module is None:
            raise KeyError(
                f"No object named '{name}' found in '{self.name}' registry"
                + f"Registry: {set(self._module_dict.keys())}"
            )
        return module

    def __contains__(self, name: str) -> bool:
        return name in self._module_dict

    def register(self, name: str | None = None, config: Dataclass | None = None) -> Callable[[C], C]:
        """Decorator to register a module in the registry.

        Parameters
        ----------
        name : str, optional
            The name to register the module under. If None, the module's
            `__name__` attribute will be used. Defaults to None.
        config : Dataclass, optional
            The OmegaConf-compatible dataclass to be used for configuration.
            If provided, the module's `__init__` method will be wrapped to
            automatically merge a default configuration created from this
            dataclass with the configuration passed during instantiation.
            Defaults to None.

        Returns
        -------
        Callable[[C], C]
            A decorator that registers the module.
        """

        def decorator(module_to_register: C) -> C:
            # Pass the module, explicit name (if any), and config class
            # to the internal registration method.
            return self._do_register(module_to_register, name_override=name, config_class=config)

        return decorator

    def _do_register(self, module: C, name_override: str | None, config_class: Dataclass | None) -> C:
        """Registers the module and optionally wraps its __init__ for config handling.

        This method is called internally by the `register` decorator. It handles
        the actual registration of the module into the `_module_dict` and,
        if a `config_class` is provided, wraps the module's `__init__` method
        to merge a default configuration with any user-provided configuration.

        Parameters
        ----------
        module : C
            The class (module) to register. It is expected to be a type.
        name_override : str or None
            If provided, this name is used for registration. Otherwise,
            the `__name__` attribute of the `module` is used as the
            registration key.
        config_class : Dataclass or None
            An OmegaConf-compatible dataclass (i.e., a type). If provided,
            the `__init__` method of the `module` will be wrapped. This
            wrapper will instantiate a default configuration object using
            `OmegaConf.structured(config_class)` and merge it with the
            `config` argument passed to the `module`'s `__init__` during
            its instantiation. The user-provided `config` takes precedence.

        Returns
        -------
        C
            The registered module. If `config_class` was provided, the
            module's `__init__` method is now a wrapped version.

        Raises
        ------
        AssertionError
            If a module with the determined name (either `name_override` or
            `module.__name__`) is already registered in this registry.
        """
        # Determine the actual name to use for registration
        name = name_override if name_override is not None else module.__name__

        # add module to registry
        assert name not in self._module_dict, (
            f"An object named '{name}' was already registered in '{self.name}' registry!"
        )
        self._module_dict[name] = module

        # wrap the config
        if config_class is not None:
            original_init = module.__init__

            @wraps(original_init)
            def wrapped_init(instance, config: Any, *args, **kwargs):
                # create a default configuration from the provided config dataclass
                merged_config = merge_base_config(config, config_class)
                return original_init(instance, merged_config, *args, **kwargs)

            module.__init__ = wrapped_init

        return module


def merge_base_config(config: Any, config_class: Dataclass) -> Any:
    assert hasattr(config_class, "_registry_"), "Config class must have a '_registry_' class attribute"
    base_config = OmegaConf.structured(config_class)
    OmegaConf.set_struct(base_config, False)
    base_config._registry_ = config_class._registry_
    OmegaConf.set_struct(base_config, True)
    merged_config = OmegaConf.merge(base_config, config)
    return merged_config


# data
DATAMODULE = Registry("datamodule")
DATASET = Registry("dataset")
TRANSFORM = Registry("transform")
PRIOR_DISTRIBUTION = Registry("prior_distribution")
TIME_DISTRIBUTION = Registry("time_distribution")
INTERPOLANT = Registry("interpolant")

# model
CFM = Registry("cfm")
MODEL = Registry("model")
