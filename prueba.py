import xml.etree.ElementTree as ET

# Cargar el archivo XML
xml_file = "voyconsciente-blog.xml"  # Asegúrate de usar la ruta correcta
ns = {'atom': 'http://www.w3.org/2005/Atom'}
tree = ET.parse(xml_file)
root = tree.getroot()

# Contar cuántas entradas son realmente de tipo "post"
posts = [
    entry for entry in root.findall('atom:entry', ns)
    if any(cat.get('term') == 'http://schemas.google.com/blogger/2008/kind#post' for cat in entry.findall('atom:category', ns))
]

# Mostrar cuántos posts hay
print(f"Número total de artículos detectados: {len(posts)}")
