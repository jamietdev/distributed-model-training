import bisect
import hashlib

class HashRing:
    def __init__(self, num_weights, num_virtual_servers=2):
        self.num_virtual_servers = num_virtual_servers
        self.ring = {} # maps position on ring -> server id (which server lives there)
        self.sorted_keys = []
        self.servers = set() 
        self.num_weights = num_weights
        # store position -> weights? 

    def hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)

    def add_server(self, server_id: str):
        for i in range(self.num_virtual_servers):
            position = self.hash(f"{server_id}#v{i}")
            self.ring[position] = server_id
            bisect.insort(self.sorted_keys, position)

        self.servers.add(server_id) 
    
    # gets the server this weight idx is mapped to
    import bisect

    def get_server(self, weight_index: int) -> str:
        if not self.ring:
            raise ValueError("No servers in ring")

        weight_pos = self.hash(str(weight_index))
        keys = self.sorted_keys

        idx = bisect.bisect_left(keys, weight_pos)

        if idx == len(keys):
            idx = 0  # wrap around

        return self.ring[keys[idx]]
    
    def build_weight_map(self, num_weights: int) -> dict[int, str]:
        # returns dict of weight index -> server id for weight indices 0 to num_weights - 1
        # called once at start up 
        weightMap = {}
        for i in range(num_weights):
            server = self.get_server(i)
            weightMap[i] = server

        return weightMap
                
    def get_weightIdxs_for_specific_server(self, server_id: str):
        weightMap = self.build_weight_map(self.num_weights)
        return [k for k, v in weightMap.items() if v == server_id]

    def weightIdxs_for_all_servers(self, num_weights: int) -> dict[str, list[int]]:
        weight_map = self.build_weight_map(num_weights)
        keys_by_server = {}

        for k, server_id in weight_map.items():
            if server_id not in keys_by_server:
                keys_by_server[server_id] = []
            keys_by_server[server_id].append(k)

        return keys_by_server

    def remove_server(self, server_id: str) -> None:
        for i in range(self.num_virtual_servers):
            position = self.hash(f"{server_id}#v{i}")
            del self.ring[position]
            self.sorted_keys.remove(position)

        self.servers.remove(server_id)

        return