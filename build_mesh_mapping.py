# build_mesh_mapping.py
import xml.etree.ElementTree as ET
from pathlib import Path

def build_mesh_mapping(xml_path: Path) -> dict:
    """从MeSH描述符XML文件中提取UI到名称的映射。"""
    mesh_id_to_name = {}
    # 使用迭代器逐条处理，避免一次性加载巨大的XML文件
    for event, elem in ET.iterparse(xml_path, events=('end',)):
        # 当我们解析完一个DescriptorRecord标签时，进行处理
        if elem.tag == 'DescriptorRecord':
            # 提取DescriptorUI (MeSH ID)
            ui_elem = elem.find('DescriptorUI')
            # 提取DescriptorName下的String (术语名称)
            name_elem = elem.find('.//DescriptorName/String')
            if ui_elem is not None and name_elem is not None:
                mesh_id = ui_elem.text
                name = name_elem.text
                mesh_id_to_name[mesh_id] = name
            # 释放当前元素，节省内存
            elem.clear()
    print(f"成功构建映射表，包含 {len(mesh_id_to_name)} 个术语。")
    return mesh_id_to_name

if __name__ == "__main__":
    # 将'desc2025.xml'替换为你下载的实际文件名
    xml_file = Path("desc2025.xml")
    mapping = build_mesh_mapping(xml_file)
    # 快速验证：查看前5个条目
    for i, (mid, name) in enumerate(mapping.items()):
        if i >= 5: break
        print(f"{mid} -> {name}")