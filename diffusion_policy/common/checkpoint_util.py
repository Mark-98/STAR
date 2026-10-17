from typing import Optional, Dict
import os
import string

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()

    def _default_monitor_value(self) -> float:
        if self.mode == 'max':
            return float('-inf')
        return float('inf')

    def _fill_missing_format_keys(self, data: Dict[str, float]) -> Dict[str, float]:
        safe_data = dict(data)
        if self.monitor_key not in safe_data:
            safe_data[self.monitor_key] = self._default_monitor_value()

        formatter = string.Formatter()
        for _, field_name, _, _ in formatter.parse(self.format_str):
            if field_name and (field_name not in safe_data):
                safe_data[field_name] = 0.0
        return safe_data

    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        safe_data = self._fill_missing_format_keys(data)
        value = safe_data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**safe_data))

        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path

        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path
