"""
Configuration file for PROJECT VERTEX
"""
import json
import os

class Config:
    def __init__(self, config_file=None):
        # Resolve config path relative to this file so the app can be launched
        # from any working directory and still find / write config.json correctly.
        if config_file is None:
            import os as _os
            config_file = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "config.json")
        self.config_file = config_file
        self.default_config = {
            "display": {
                "width": 1200,
                "height": 900,
                "fov": 45,
                "near": 0.1,
                "far": 50.0
            },
            "camera": {
                "initial_z": -5,
                "zoom_min": -15,
                "zoom_max": -2
            },
            "controls": {
                "rotation_sensitivity": 0.5,
                "zoom_sensitivity": 0.1,
                "smoothing_factor": 0.1,
                "pinch_threshold": 40,
                "zoom_base_distance": 200,
                "zoom_exponent": 1.5,
                "rotation_velocity_multiplier": 0.3,
                "swipe_enabled": True,
                "fist_enabled": True,
                "pointing_enabled": True
            },
            "rendering": {
                "wireframe": False,
                "show_grid": True,
                "show_axes": True,
                "show_hud": True,
                "shape_scale": 1.0,
                "color_index": 0,
                "background_color": [0.04, 0.04, 0.10, 1.0],
                "shape_color": [0.0, 1.0, 1.0],
                "grid_color": [0.3, 0.3, 0.3],
                "axes_color": [1.0, 1.0, 1.0]
            },
            "hand_sensor": {
                "detection_confidence": 0.5,
                "tracking_confidence": 0.5
            },
            "shapes": {
                "default": "cube",
                "available": [
                    "cube", "pyramid", "sphere", "cylinder", "torus",
                    "octahedron", "cone", "diamond", "icosahedron", "torus_knot"
                ]
            }
        }
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    config = self._merge_dicts(self.default_config, user_config)
                    return config
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
                return self.default_config.copy()
        else:
            self.save_config(self.default_config)
            return self.default_config.copy()
    
    def save_config(self, config=None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _merge_dicts(self, default, user):
        """Recursively merge user config into default"""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, *keys):
        """Get nested config value"""
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            # Return None if key doesn't exist or path is invalid
            return None
    
    def set(self, *keys, value):
        """Set nested config value"""
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self.save_config()

    def __repr__(self):
        import json
        return f"Config(file={self.config_file!r})\n{json.dumps(self.config, indent=2)}"

