# extraer_y_normalizar.py
from app import app, db, Reflexion
from bs4 import BeautifulSoup
import json

def limpiar_y_estructurar_html(reflexion):
    contenido = reflexion['contenido']
    soup = BeautifulSoup(contenido, 'html.parser')
    
    # Eliminar atributos innecesarios (como style) pero mantener la estructura
    for tag in soup.find_all(True):
        # Conservar solo atributos esenciales en imágenes
        if tag.name == 'img':
            src = tag.get('src', '')
            alt = tag.get('alt', '')
            tag.attrs = {'class': 'reflexion-image', 'src': src, 'alt': alt}
        else:
            tag.attrs = {}  # Eliminar todos los atributos excepto en imágenes
    
    # Crear una nueva estructura
    nuevo_contenido = []
    
    # Agregar h1 basado en el título del artículo si no hay h1
    if not soup.find('h1'):
        nuevo_contenido.append(f'<h1 class="reflexion-title reflexion-title-h1">{reflexion["titulo"]}</h1>')
    
    # Procesar cada elemento del contenido original
    for elemento in soup.find_all(recursive=False):  # Solo elementos de nivel superior
        if elemento.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            nivel = elemento.name
            # Mantener formato interno (b, i, u)
            contenido_h = str(elemento).replace(f'<{nivel}>', '').replace(f'</{nivel}>', '').strip()
            nuevo_contenido.append(f'<{nivel} class="reflexion-title reflexion-title-{nivel}">{contenido_h}</{nivel}>')
        
        elif elemento.name == 'img':
            nuevo_contenido.append(str(elemento))  # Ya tiene clase reflexion-image
        
        elif elemento.name in ['ul', 'ol']:
            tipo_lista = elemento.name
            lista_items = []
            for li in elemento.find_all('li', recursive=False):
                # Preservar formato interno
                contenido_li = str(li).replace('<li>', '').replace('</li>', '').strip()
                lista_items.append(f'<li class="reflexion-list-item">{contenido_li}</li>')
            nuevo_contenido.append(f'<{tipo_lista} class="reflexion-list reflexion-list-{tipo_lista}">\n' + '\n'.join(lista_items) + f'\n</{tipo_lista}>')
        
        elif elemento.name == 'p' or elemento.name in ['div', 'span'] or elemento.name is None:
            # Convertir todo a párrafos, preservando formato interno
            texto = elemento.get_text(strip=True) if elemento.name else elemento.strip()
            if texto and len(texto) > 10:
                # Mantener b, i, u si existen
                contenido_p = str(elemento).replace(f'<{elemento.name}>', '').replace(f'</{elemento.name}>', '').strip() if elemento.name else texto
                nuevo_contenido.append(f'<p class="reflexion-paragraph">{contenido_p}</p>')
    
    # Generar h2 si no hay y el contenido lo requiere
    tiene_h2 = any('reflexion-title-h2' in linea for linea in nuevo_contenido)
    if not tiene_h2 and len(soup.get_text(strip=True)) > 500:
        texto_completo = soup.get_text(separator='\n', strip=True)
        lineas = [l.strip() for l in texto_completo.split('\n') if l.strip()]
        insertados = 0
        for i, linea in enumerate(lineas):
            if len(linea) > 50 and linea.endswith(('.', '!', '?')) and insertados < 2:  # Limitar a 2 h2 automáticos
                nuevo_contenido.insert(i + insertados, f'<h2 class="reflexion-title reflexion-title-h2">{linea}</h2>')
                insertados += 1
    
    return '\n'.join(nuevo_contenido)

# Extraer reflexiones de la base de datos
reflexiones_db = []
with app.app_context():
    reflexiones = Reflexion.query.all()
    for r in reflexiones:
        reflexiones_db.append({
            'id': r.id,
            'titulo': r.titulo,
            'contenido': r.contenido,
            'fecha': r.fecha,
            'categoria': r.categoria,
            'imagen': r.imagen
        })

# Normalizar las reflexiones
reflexiones_limpias = []
for reflexion in reflexiones_db:
    try:
        contenido_limpio = limpiar_y_estructurar_html(reflexion)
        reflexion_limpia = reflexion.copy()
        reflexion_limpia['contenido'] = contenido_limpio
        reflexiones_limpias.append(reflexion_limpia)
    except Exception as e:
        print(f"Error al procesar reflexión ID {reflexion['id']}: {e}")

# Mostrar resultado
for r in reflexiones_limpias:
    print(f"ID: {r['id']}")
    print(f"Título: {r['titulo']}")
    print("Contenido limpio:")
    print(r['contenido'])
    print("\n---\n")

# Guardar en un archivo JSON
with open('reflexiones_limpias.json', 'w', encoding='utf-8') as f:
    json.dump(reflexiones_limpias, f, ensure_ascii=False, indent=2)

print("Reflexiones extraídas y normalizadas guardadas en 'reflexiones_limpias.json'.")