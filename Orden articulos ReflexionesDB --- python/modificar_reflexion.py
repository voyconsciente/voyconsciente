from app import app, db, Reflexion

with app.app_context():
    id_a_modificar = 1  # Cambia este ID por el de la reflexión que quieras modificar
    reflexion = Reflexion.query.get(id_a_modificar)

    if reflexion:
        reflexion.titulo = "Nuevo título actualizado"
        reflexion.contenido = "Este es el contenido actualizado de la reflexión."
        db.session.commit()
        print("Artículo modificado con éxito.")
    else:
        print("No se encontró el artículo con ese ID.")
