from typing import Optional, Tuple, List, Dict

class MockedDatabaseSchema():
    def __init__(self, schema_name: str, full_schema: Optional[bool] = False):
        pass

    @property
    def all(self) -> List[str]:
        return ["Mock Schema"]

    @all.setter
    def all(self, new: list):
        pass
        
    @property
    def schema_name(self) -> str:
        pass
    
    @schema_name.setter
    def schema_name(self, new: str):
        pass
        
    @property
    def schema_tree(self) -> Dict[str, List[str]]:
        pass

    @schema_tree.setter
    def schema_tree(self, new: Dict[str, str]):
        pass
        
    @property
    def all_tables(self) -> List[str]:
        return ["Mock Table"]
    
    @all_tables.setter
    def all_tables(self, new: List[str]): 
        pass
    
    @property
    def table_count(self) -> int:
        pass
    
    @table_count.setter
    def table_count(self, new: int):
        pass

    def add_schema(self, schema: str, position: Optional[int] = None):
        pass
            
    def _get_all_tables(self, schema: Optional[str] = None) -> List[str]:
        pass
    
    def _get_table_count(self, schema: Optional[str] = None) -> int:
        pass
    
    def _table_info(self, schema: Optional[str] = None):
        pass

    def _build_schema_dict(self):
        pass

    def _show_schema_dict(self):
        pass
