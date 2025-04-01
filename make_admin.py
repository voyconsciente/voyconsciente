from app import app, db, User  # Asegúrate de que 'app' sea el nombre de tu archivo principal

def make_user_admin(email):
    with app.app_context():
        # Buscar al usuario por su correo
        user = User.query.filter_by(email=email).first()
        if user:
            user.is_admin = True
            db.session.commit()
            print(f"El usuario {email} ahora es administrador.")
        else:
            print(f"No se encontró un usuario con el correo {email}.")

if __name__ == "__main__":
    email_to_admin = "jesicadesario@hotmail.com"  # Cambia esto al correo del usuario
    make_user_admin(email_to_admin)