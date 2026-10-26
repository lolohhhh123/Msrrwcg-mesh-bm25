# mesh_translator.py
import re
from build_mesh_mapping import build_mesh_mapping
from pathlib import Path

class MeshTranslator:
    def __init__(self, xml_path: Path):
        self.mapping = build_mesh_mapping(xml_path)
        self.mesh_pattern = re.compile(r'(meshd\d+)', re.IGNORECASE)

    def replace_mesh_ids_in_text(self, text: str) -> str:
        def _replace(match):
            mesh_id = match.group(1).lower()
            return self.mapping.get(mesh_id, mesh_id)
        return self.mesh_pattern.sub(_replace, text)

if __name__ == "__main__":
    mesh_xml = Path("./data/mesh/desc2025.xml")
    translator = MeshTranslator(mesh_xml)

    # test
    raw_abstract = "The role of meshd003643 in meshd011506 regulation ..."
    readable_abstract = translator.replace_mesh_ids_in_text(raw_abstract)
    print(readable_abstract)
