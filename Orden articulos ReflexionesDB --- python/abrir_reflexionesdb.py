from app import app, db, Reflexion  # Asegúrate de importar bien estos módulos

with app.app_context():
    reflexiones = Reflexion.query.all()  # Ahora la base de datos funcionará correctamente

    for r in reflexiones:
        print(r.id, r.titulo, r.categoria, r.fecha)
