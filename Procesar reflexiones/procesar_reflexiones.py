import re
from bs4 import BeautifulSoup # type: ignore
import ast

def limpiar_html(contenido):
    soup = BeautifulSoup(contenido, 'html.parser')
    
    # Eliminar spans redundantes sin clase o estilo útil
    for span in soup.find_all('span'):
        if not span.get('class') and not span.get('style') or (span.get('style') and 'color' not in span.get('style') and 'font-weight' not in span.get('style')):
            span.unwrap()
    
    # Crear estructura base
    post_content = soup.new_tag('div', **{'class': 'post-content'})
    
    # Procesar elementos
    children = list(soup.body.children if soup.body else soup.children)
    for child in children:
        if child.name:
            if child.name == 'img':
                img_wrapper = soup.new_tag('figure')
                child['style'] = 'max-width: 100%; height: auto; margin-bottom: 12px;'
                for attr in ['data-lazy-loaded', 'class', 'sizes', 'srcset', 'height', 'width']:
                    if attr in child.attrs:
                        del child[attr]
                img_wrapper.append(child)
                post_content.append(img_wrapper)
            elif child.name in ['h1', 'h2', 'h3']:
                # Aplicar clases de título
                if child.name == 'h1':
                    child['class'] = child.get('class', []) + ['title-main']
                elif child.name == 'h2':
                    child['class'] = child.get('class', []) + ['title-sub']
                elif child.name == 'h3':
                    child['class'] = child.get('class', []) + ['title-sub-alt']
                # Eliminar estilos inline
                if 'style' in child.attrs:
                    del child['style']
                post_content.append(child)
            elif child.name == 'div':
                if not child.get('class'):
                    p = soup.new_tag('p')
                    text_content = ''.join(str(c).replace('\n', ' ').strip() for c in child.contents if c)
                    p.string = re.sub(r'\s+', ' ', text_content)
                    if 'style' in child.attrs:
                        del child['style']  # Eliminar estilos inline
                    post_content.append(p)
                else:
                    if 'tm-click-to-tweet' in child.get('class', []):
                        post_content.append(child)
                    elif any(cls in ['highlight', 'tm-ctt-text'] for cls in child.get('class', [])):
                        highlight = soup.new_tag('p', **{'class': 'highlight'})
                        highlight.extend(child.contents)
                        post_content.append(highlight)
                    elif 'separator' in child.get('class', []):
                        # Mantener imágenes en separadores como figure
                        post_content.append(child)
                    else:
                        p = soup.new_tag('p')
                        text_content = ''.join(str(c).replace('\n', ' ').strip() for c in child.contents if c)
                        p.string = re.sub(r'\s+', ' ', text_content)
                        if 'style' in child.attrs:
                            del child['style']  # Eliminar estilos inline
                        post_content.append(p)
            elif child.name in ['ul', 'ol', 'blockquote']:
                # Mantener listas y citas sin estilos inline innecesarios
                if 'style' in child.attrs:
                    del child['style']
                post_content.append(child)
            else:
                post_content.append(child)
    
    # Resaltar palabras clave
    for p in post_content.find_all('p'):
        if p.string:
            text = p.string
            text = re.sub(r'\b(libertad|responsabilidad|decisiones|acciones)\b', r'<span class="keyword">\1</span>', text, flags=re.IGNORECASE)
            p.clear()
            p.append(BeautifulSoup(text, 'html.parser'))
    
    return str(post_content).replace('\n', '').replace('  ', ' ').strip()

def escapar_comillas(cadena):
    if cadena is None:
        return None
    return cadena.replace('"', '\\"')

def procesar_reflexiones(entrada_path, salida_path):
    with open(entrada_path, 'r', encoding='utf-8') as f:
        contenido = f.read()
    tree = ast.parse(contenido)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and node.targets[0].id == 'reflexiones_data':
            reflexiones_data = ast.literal_eval(node.value)
            break
    reflexiones_optimizadas = []
    for reflexion in reflexiones_data:
        contenido_limpio = limpiar_html(reflexion['contenido'])
        reflexiones_optimizadas.append({
            'id': reflexion['id'],
            'titulo': reflexion['titulo'],
            'contenido': contenido_limpio,
            'fecha': reflexion['fecha'],
            'categoria': reflexion['categoria'],
            'imagen': reflexion['imagen']
        })
    with open(salida_path, 'w', encoding='utf-8') as f:
        f.write('reflexiones_data = [\n')
        for i, reflexion in enumerate(reflexiones_optimizadas):
            f.write('    {\n')
            f.write(f"        'id': {reflexion['id']},\n")
            f.write(f"        'titulo': \"{escapar_comillas(reflexion['titulo'])}\",\n")
            f.write(f"        'contenido': \"\"\"{escapar_comillas(reflexion['contenido'])}\"\"\",\n")
            f.write(f"        'fecha': \"{escapar_comillas(reflexion['fecha'])}\",\n")
            f.write(f"        'categoria': \"{escapar_comillas(reflexion['categoria'])}\",\n")
            f.write(f"        'imagen': {'None' if reflexion['imagen'] is None else f'\"{escapar_comillas(reflexion['imagen'])}\"'}\n")
            f.write('    }' + (',' if i < len(reflexiones_optimizadas) - 1 else '') + '\n')
        f.write(']\n')

if __name__ == "__main__":
    entrada = 'reflexiones_importadas.py'  # Ajusta según tu archivo de entrada
    salida = 'reflexiones_optimizadas.py'
    procesar_reflexiones(entrada, salida)
    print(f"Archivo optimizado generado en: {salida}")