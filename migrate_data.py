from app import app, db, Reflexion
from reflexiones_optimizadas_EDITABLE import reflexiones_data

with app.app_context():
    # Limpiar la base de datos (opcional)
    db.drop_all()
    db.create_all()

    # Insertar datos
    for r in reflexiones_data:
        reflexion = Reflexion(
            id=r['id'],
            titulo=r['titulo'],
            contenido=r['contenido'],
            fecha=r['fecha'],
            categoria=r['categoria'],
            imagen=r['imagen']
        )
        db.session.add(reflexion)
    db.session.commit()
    print("Datos migrados exitosamente!")