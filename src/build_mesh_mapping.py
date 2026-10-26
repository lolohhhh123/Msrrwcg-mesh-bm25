# build_mesh_mapping.py
import xml.etree.ElementTree as ET
from pathlib import Path

def build_mesh_mapping(xml_path: Path) -> dict:
    mesh_id_to_name = {}
    for event, elem in ET.iterparse(xml_path, events=('end',)):
        if elem.tag == 'DescriptorRecord':
            ui_elem = elem.find('DescriptorUI')
            name_elem = elem.find('.//DescriptorName/String')
            if ui_elem is not None and name_elem is not None:
                mesh_id = ui_elem.text
                name = name_elem.text
                mesh_id_to_name[mesh_id] = name
            elem.clear()
    print(f"Finish mapping，included {len(mesh_id_to_name)} MeSH terms.")
    return mesh_id_to_name

if __name__ == "__main__":
    xml_file = Path("desc2025.xml")
    mapping = build_mesh_mapping(xml_file)
    # Test
    for i, (mid, name) in enumerate(mapping.items()):
        if i >= 5: break
        print(f"{mid} -> {name}")
