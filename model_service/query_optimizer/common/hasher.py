import hashlib
import json
import numpy as np


class FeatureHasher:
    def __init__(self, strict_keys: list, fuzzy_keys: list, tolerance: float = 0.05):
        """
        strict_keys: список полей для точного совпадения.
        fuzzy_keys: список полей для поиска подобия.
        tolerance: порог отклонения (0.05 = 5%).
        """
        self.strict_keys = strict_keys
        self.fuzzy_keys = fuzzy_keys
        # Коэффициент для квантования: log(1.05)
        self.step = np.log(1 + tolerance)

    def compute_hash(self, data: dict) -> str:
        hash_payload = {}

        # 1. Жесткие ключи (целые числа) - должны совпадать 1 в 1
        for key in self.strict_keys:
            val = data.get(key, 0)
            hash_payload[key] = int(val) if val is not None else 0

        # 2. Гибкие ключи (float) - подобие в пределах 5%
        for key in self.fuzzy_keys:
            val = float(data.get(key, 0) or 0)
            if val > 0:
                # Квантование: переводим значение в "номер корзины"
                # Каждая корзина имеет ширину ровно 5%
                bucket = int(np.log(val + 1) / self.step)
            else:
                bucket = 0
            hash_payload[key] = bucket

        # 3. Стабильный хэш
        dump = json.dumps(hash_payload, sort_keys=True)
        return hashlib.md5(dump.encode()).hexdigest()
