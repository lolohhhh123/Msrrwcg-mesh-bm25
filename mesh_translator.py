# mesh_translator.py
import re
from build_mesh_mapping import build_mesh_mapping
from pathlib import Path

class MeshTranslator:
    def __init__(self, xml_path: Path):
        # 加载由步骤二生成的映射表
        self.mapping = build_mesh_mapping(xml_path)
        # 预编译正则表达式，用于高效匹配文本中的meshd ID
        self.mesh_pattern = re.compile(r'(meshd\d+)', re.IGNORECASE)

    def replace_mesh_ids_in_text(self, text: str) -> str:
        """将文本中的所有meshd ID替换为人类可读的术语。"""
        def _replace(match):
            mesh_id = match.group(1).lower()
            # 从映射表中获取术语，如果找不到则返回原始ID
            return self.mapping.get(mesh_id, mesh_id)
        return self.mesh_pattern.sub(_replace, text)

# 在你的数据处理脚本中使用这个类
if __name__ == "__main__":
    # 假设下载的MeSH XML文件路径
    mesh_xml = Path("./data/mesh/desc2025.xml")
    translator = MeshTranslator(mesh_xml)

    # 模拟从RELISH数据集中读取的一条记录
    raw_abstract = "The role of meshd003643 in meshd011506 regulation ..."
    readable_abstract = translator.replace_mesh_ids_in_text(raw_abstract)
    print(readable_abstract)