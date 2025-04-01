from app import app, db, Reflexion

with app.app_context():
    id_a_modificar = 1  # ID de la reflexión que modificaste
    reflexion = Reflexion.query.get(id_a_modificar)

    if reflexion:
        reflexion.titulo = "Título anterior"
        reflexion.contenido = "Este era el contenido original de la reflexión."
        db.session.commit()
        print("Artículo restaurado con éxito.")
    else:
        print("No se encontró el artículo con ese ID.")
