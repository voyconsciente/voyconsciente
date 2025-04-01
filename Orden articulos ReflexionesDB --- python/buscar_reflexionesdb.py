from app import app, db, Reflexion  # Importa la aplicación y el modelo

query = "felicidad"  # Puedes cambiarlo por lo que quieras buscar

with app.app_context():  # Necesario para acceder a la base de datos
    resultados = Reflexion.query.filter(
        Reflexion.titulo.ilike(f'%{query}%') | Reflexion.contenido.ilike(f'%{query}%')
    ).all()

    for r in resultados:
        print(r.titulo)
