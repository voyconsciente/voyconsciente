import sqlite3
from app import db, Subscriber  # Asegúrate de importar tu app correctamente

# Conectar a subscribers.db
old_conn = sqlite3.connect('/Users/sebastianredigonda/Desktop/voy_consciente/subscribers.db')
old_cursor = old_conn.execute('SELECT email, subscription_date FROM subscribers')
subscribers = old_cursor.fetchall()
old_conn.close()

# Migrar a basededatos.db
with app.app_context():
    for email, subscription_date in subscribers:
        subscriber = Subscriber(email=email, subscription_date=subscription_date)
        db.session.add(subscriber)
    db.session.commit()
    print("Suscriptores migrados a basededatos.db")