"""Memory tape allocation and register management for Brainfuck compilation."""

from typing import Dict, List, Optional, Set


class TapeMemoryManager:
    """Manages allocation of variables, dual-rail registers, and scratch cells on the BF tape."""

    def __init__(self):
        self._cell_map: Dict[str, int] = {}
        self._allocated_cells: Set[int] = set()
        self._free_scratch_cells: List[int] = []
        self._current_ptr: int = 0
        self._max_ptr: int = 0

    @property
    def current_ptr(self) -> int:
        return self._current_ptr

    @property
    def max_cell_used(self) -> int:
        return self._max_ptr

    @property
    def cell_map(self) -> Dict[str, int]:
        return dict(self._cell_map)

    def allocate(self, name: str, fixed_cell: Optional[int] = None) -> int:
        """Allocate a named cell on the tape."""
        if name in self._cell_map:
            return self._cell_map[name]

        if fixed_cell is not None:
            cell = fixed_cell
        else:
            # Find lowest non-allocated cell
            cell = 0
            while cell in self._allocated_cells:
                cell += 1

        self._allocated_cells.add(cell)
        self._cell_map[name] = cell
        self._max_ptr = max(self._max_ptr, cell)
        return cell

    def allocate_dual_rail(self, base_name: str) -> tuple[int, int]:
        """Allocate positive and negative cell pair for signed dual-rail variable."""
        pos = self.allocate(f"{base_name}_pos")
        neg = self.allocate(f"{base_name}_neg")
        return pos, neg

    def allocate_scratch(self, name_prefix: str = "temp") -> str:
        """Allocate an anonymous scratch cell."""
        idx = 0
        while True:
            name = f"__{name_prefix}_{idx}"
            if name not in self._cell_map:
                self.allocate(name)
                return name
            idx += 1

    def free(self, name: str) -> None:
        """Free a named variable and allow its cell to be reused."""
        if name in self._cell_map:
            cell = self._cell_map.pop(name)
            self._allocated_cells.discard(cell)
            self._free_scratch_cells.append(cell)

    def get_cell(self, name: str) -> int:
        """Retrieve cell index for a variable."""
        if name not in self._cell_map:
            raise KeyError(f"Variable '{name}' is not allocated on the tape.")
        return self._cell_map[name]

    def move_ptr_to(self, target_cell: int) -> str:
        """Generate Brainfuck pointer movement string from current_ptr to target_cell."""
        delta = target_cell - self._current_ptr
        self._current_ptr = target_cell
        if delta > 0:
            return ">" * delta
        elif delta < 0:
            return "<" * (-delta)
        return ""

    def move_to(self, name: str) -> str:
        """Generate Brainfuck code to navigate pointer to variable 'name'."""
        cell = self.get_cell(name)
        return self.move_ptr_to(cell)
