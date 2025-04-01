from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, make_response, session
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from flask_assets import Environment, Bundle
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from onesignal_sdk.client import Client as OneSignalClient
from PIL import Image
from weasyprint import HTML
from apscheduler.schedulers.background import BackgroundScheduler
import io
import datetime
import random
import os
import sqlite3
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.oauth2 import service_account
import google.generativeai as genai
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv  # Importar python-dotenv

# Cargar variables de entorno desde .env
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')  # Clave secreta desde .env
if not app.secret_key:
    raise ValueError("Error: No se encontró SECRET_KEY en las variables de entorno.")
print(f"Carpeta estática: {app.static_folder}")

# Configuración de Google Analytics
GA_CREDENTIALS_PATH = os.getenv('GA_CREDENTIALS_PATH', os.path.join(app.root_path, 'voy-consciente-analytics.json'))
GA_PROPERTY_ID = os.getenv('GA_PROPERTY_ID', '480922494')  # ID de propiedad desde .env o valor por defecto
GA_FLOW_ID = os.getenv('GA_FLOW_ID', '10343079148')  # ID del flujo desde .env o valor por defecto
GA_FLOW_NAME = os.getenv('GA_FLOW_NAME', 'Voy Consciente')  # Nombre del flujo desde .env o valor por defecto
GA_FLOW_URL = os.getenv('GA_FLOW_URL', 'https://192.168.0.213:5001')  # URL del flujo desde .env o valor por defecto

credentials = service_account.Credentials.from_service_account_file(GA_CREDENTIALS_PATH)
analytics_client = BetaAnalyticsDataClient(credentials=credentials)

# Configuración de Gemini API con variable de entorno
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Error: No se encontró la clave API de Gemini en las variables de entorno.")
else:
    print(f"Clave API de Gemini cargada: {api_key[:5]}...")
genai.configure(api_key=api_key)

# Configuración de Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configuración de la base de datos desde variable de entorno
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:////Users/sebastianredigonda/Desktop/voy_consciente/basededatos.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configuración de caché
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configuración de Flask-Mail con variables de entorno
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')
if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
    raise ValueError("Error: Faltan MAIL_USERNAME o MAIL_PASSWORD en las variables de entorno.")
mail = Mail(app)

# Configuración de Flask-Assets
assets = Environment(app)
assets.url = '/static'
assets.directory = os.path.join(app.root_path, 'static')
css = Bundle('css/style.css', filters='cssmin', output='gen/style.min.css')
assets.register('css_all', css)

# Configuración de OneSignal con variables de entorno
onesignal_client = OneSignalClient(
    app_id=os.getenv("ONESIGNAL_APP_ID"),
    rest_api_key=os.getenv("ONESIGNAL_REST_API_KEY")
)
if not onesignal_client.app_id or not onesignal_client.rest_api_key:
    raise ValueError("Error: Faltan ONESIGNAL_APP_ID o ONESIGNAL_REST_API_KEY en las variables de entorno.")

# Función para enviar reflexiones programadas
def send_weekly_reflection():
    with app.app_context():
        subscribers = Subscriber.query.all()
        for subscriber in subscribers:
            # Filtrar reflexiones según categorías preferidas
            categories = subscriber.preferred_categories.split(',')
            if 'all' in categories:
                reflection = Reflexion.query.order_by(db.func.random()).first()
            else:
                reflection = Reflexion.query.filter(Reflexion.categoria.in_(categories)).order_by(db.func.random()).first()
            
            if not reflection:
                print(f"No hay reflexiones para {subscriber.email} en {categories}")
                continue

            # Enviar correo
            msg = Message('Tu Reflexión Personalizada - Voy Consciente', recipients=[subscriber.email])
            msg.body = f'Reflexión: "{reflection.contenido}"'
            msg.html = f"""
            <html>
                <body style="font-family: Arial, sans-serif; color: #333; background-color: #f9f9f9; padding: 20px;">
                    <div style="max-width: 600px; margin: 0 auto; background-color: #fff; border-radius: 10px; padding: 20px;">
                        <h1 style="color: #4CAF50; text-align: center;">Tu Reflexión Personalizada</h1>
                        <p style="font-style: italic;">"{reflection.contenido}"</p>
                        <p style="text-align: center;">- {reflection.titulo}</p>
                        <p style="text-align: center;"><a href="{GA_FLOW_URL}/reflexion/{reflection.id}" style="color: #4CAF50;">Leer más</a></p>
                        <p style="text-align: center;"><a href="{GA_FLOW_URL}/preferencias" style="color: #4CAF50;">Cambiar preferencias</a></p>
                    </div>
                </body>
            </html>
            """
            try:
                mail.send(msg)
                print(f'Correo enviado a {subscriber.email}')
            except Exception as e:
                print(f'Error enviando a {subscriber.email}: {str(e)}')

            # Enviar notificación push si está habilitada
            if subscriber.push_enabled and subscriber.onesignal_player_id:
                notification = {
                    "contents": {"en": f"{reflection.titulo}: {reflection.contenido[:50]}..."},
                    "include_player_ids": [subscriber.onesignal_player_id]
                }
                try:
                    response = onesignal_client.notification_create(notification)
                    print(f'Notificación push enviada a {subscriber.email}: {response.body}')
                except Exception as e:
                    print(f'Error enviando push a {subscriber.email}: {str(e)}')

# Modelo de Usuario
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Modelo de Reflexión
class Reflexion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    titulo = db.Column(db.String(200), nullable=False)
    contenido = db.Column(db.Text, nullable=False)
    fecha = db.Column(db.String(10))
    categoria = db.Column(db.String(50))
    imagen = db.Column(db.String(200))

# Modelo de Favoritos
favorite = db.Table('favorite',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('reflexion_id', db.Integer, db.ForeignKey('reflexion.id'), primary_key=True)
)

User.favorite_reflexiones = db.relationship('Reflexion', secondary=favorite, backref=db.backref('favorited_by', lazy='dynamic'))

# Modelo de Suscriptores
class Subscriber(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    subscription_date = db.Column(db.String(30), nullable=False)
    preferred_categories = db.Column(db.String(200), default='all')
    frequency = db.Column(db.String(20), default='weekly')
    push_enabled = db.Column(db.Boolean, default=False)
    onesignal_player_id = db.Column(db.String(50))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Función para comprimir imágenes
def compress_image(image_path, output_path, max_width=600, quality=85):
    try:
        with Image.open(image_path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            width, height = img.size
            if width > max_width:
                new_height = int((max_width / width) * height)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
        print(f"Imagen comprimida: {output_path}")
    except Exception as e:
        print(f"Error al comprimir imagen: {e}")
        
      # OneSignal
@app.route('/update_push', methods=['POST'])
def update_push():
    data = request.get_json()
    email = data.get('email')
    player_id = data.get('player_id')
    if email:
        subscriber = Subscriber.query.filter_by(email=email).first()
        if subscriber:
            subscriber.onesignal_player_id = player_id
            db.session.commit()
            return jsonify({'status': 'success'})
    return jsonify({'status': 'error'})

# Suscripción
@app.route('/suscribirse', methods=['GET', 'POST'])
def suscribirse():
    if request.method == 'POST':
        email = request.form.get('email')
        categories = ','.join(request.form.getlist('categories')) or 'all'
        frequency = request.form.get('frequency', 'weekly')
        push_enabled = bool(request.form.get('push_enabled'))

        if not email:
            flash('Por favor, ingresa un correo válido.', 'error')
            return redirect(url_for('home'))

        try:
            subscriber = Subscriber(
                email=email,
                subscription_date=datetime.datetime.now().isoformat(),
                preferred_categories=categories,
                frequency=frequency,
                push_enabled=push_enabled
            )
            db.session.add(subscriber)
            db.session.commit()

            msg = Message('Bienvenido a Voy Consciente', recipients=[email])
            msg.body = '¡Gracias por unirte! Personalizaste tus reflexiones.'
            msg.html = f"""
            <html>
                <body style="font-family: Arial, sans-serif; color: #333; background-color: #f9f9f9; padding: 20px;">
                    <div style="max-width: 600px; margin: 0 auto; background-color: #fff; border-radius: 10px; padding: 20px;">
                        <h1 style="color: #4CAF50; text-align: center;">¡Bienvenido!</h1>
                        <p>Gracias por unirte. Tus preferencias: {categories}, {frequency}.</p>
                        <p style="text-align: center;"><a href="{GA_FLOW_URL}/preferencias" style="color: #4CAF50;">Actualizar preferencias</a></p>
                    </div>
                </body>
            </html>
            """
            mail.send(msg)
            flash('¡Suscripción exitosa! Revisa tu correo.', 'success')
        except IntegrityError:
            db.session.rollback()
            flash('Este correo ya está suscrito.', 'error')
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('home'))

    categories = ['motivación', 'mindfulness', 'crecimiento', 'bienestar']
    return render_template('suscribirse.html', categories=categories)

# Rutas legales
@app.route('/terminos-condiciones')
def terminos_condiciones():
    return render_template('terminos_condiciones.html', año_actual=datetime.datetime.now().year)

@app.route('/politica-privacidad')
def politica_privacidad():
    return render_template('politica_privacidad.html', año_actual=datetime.datetime.now().year)

# Analíticas
@app.route('/analisis')
def analisis():
    request_ga = {
        'property': f'properties/{GA_PROPERTY_ID}',
        'date_ranges': [{'start_date': '7daysAgo', 'end_date': 'today'}],
        'dimensions': [{'name': 'date'}],
        'metrics': [
            {'name': 'activeUsers'},
            {'name': 'sessions'},
            {'name': 'averageSessionDuration'},
            {'name': 'screenPageViews'}
        ]
    }
    try:
        response = analytics_client.run_report(request_ga)
        metrics_data = []
        for row in response.rows:
            metrics_data.append({
                'date': row.dimension_values[0].value,
                'active_users': row.metric_values[0].value,
                'sessions': row.metric_values[1].value,
                'avg_session_duration': float(row.metric_values[2].value) / 60,
                'page_views': row.metric_values[3].value
            })
        return render_template('analisis.html', 
                              metrics=metrics_data, 
                              flow_name=GA_FLOW_NAME, 
                              flow_url=GA_FLOW_URL, 
                              flow_id=GA_FLOW_ID)
    except Exception as e:
        print(f"Error al obtener métricas de Google Analytics: {e}")
        flash('No se pudieron cargar las métricas en este momento.', 'error')
        return render_template('analisis.html', 
                              metrics=[], 
                              flow_name=GA_FLOW_NAME, 
                              flow_url=GA_FLOW_URL, 
                              flow_id=GA_FLOW_ID)

# Rutas estáticas
@app.route('/static/js/main.js')
def serve_main_js():
    file_path = os.path.join(app.static_folder, 'js/main.js')
    print(f"Intentando servir: {file_path}")
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, 'js/main.js')
    else:
        print(f"Archivo no encontrado en: {file_path}")
        return "Archivo no encontrado", 404

@app.route('/OneSignalSDK.sw.js')
def serve_onesignal_sw():
    file_path = os.path.join(app.static_folder, 'OneSignalSDK.sw.js')
    print(f"Intentando servir: {file_path}")
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, 'OneSignalSDK.sw.js', mimetype='application/javascript')
    else:
        print(f"Archivo no encontrado en: {file_path}")
        return "Archivo no encontrado", 404

@app.route('/OneSignalSDKWorker.js')
def serve_onesignal_worker():
    file_path = os.path.join(app.static_folder, 'OneSignalSDKWorker.js')
    print(f"Intentando servir: {file_path}")
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, 'OneSignalSDKWorker.js', mimetype='application/javascript')
    else:
        print(f"Archivo no encontrado en: {file_path}")
        return "Archivo no encontrado", 404

@app.route('/test-static/<path:filename>')
def test_static(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception as e:
        print(f"Error sirviendo archivo estático {filename}: {e}")
        return "Archivo no encontrado", 404

@app.route('/preferencias', methods=['GET', 'POST'])
def preferencias():
    if request.method == 'POST':
        email = request.form.get('email')
        subscriber = Subscriber.query.filter_by(email=email).first()
        if subscriber:
            subscriber.preferred_categories = ','.join(request.form.getlist('categories')) or 'all'
            subscriber.frequency = request.form.get('frequency', 'weekly')
            subscriber.push_enabled = bool(request.form.get('push_enabled'))
            db.session.commit()
            flash('Preferencias actualizadas.', 'success')
        else:
            flash('Correo no encontrado.', 'error')
        return redirect(url_for('preferencias'))
    
    categories = ['motivación', 'mindfulness', 'crecimiento', 'bienestar']
    return render_template('preferencias.html', categories=categories)

# Rutas principales
@app.route('/')
def home():
    return render_template('index.html', año_actual=datetime.datetime.now().year)

@app.route('/sobre-nosotros')
def sobre_nosotros():
    return render_template('sobre_nosotros.html')

@app.route('/reflexiones', defaults={'page': 1})
@app.route('/reflexiones/page/<int:page>')
@cache.cached(timeout=300, query_string=True)
def mostrar_reflexiones(page):
    per_page = 20
    reflexiones = Reflexion.query.paginate(page=page, per_page=per_page, error_out=False)
    print(f"Página {page}: {len(reflexiones.items)} reflexiones enviadas de {reflexiones.total} totales")
    return render_template('reflexiones.html', reflexiones=reflexiones.items, pagination=reflexiones)

@app.route('/reflexiones/<categoria>', defaults={'page': 1})
@app.route('/reflexiones/<categoria>/page/<int:page>')
@cache.cached(timeout=300, query_string=True)
def reflexiones_por_categoria(categoria, page):
    per_page = 20
    reflexiones = Reflexion.query.filter_by(categoria=categoria).paginate(page=page, per_page=per_page, error_out=False)
    print(f"Página {page}, Categoría {categoria}: {len(reflexiones.items)} reflexiones enviadas de {reflexiones.total} totales")
    return render_template('reflexiones.html', reflexiones=reflexiones.items, pagination=reflexiones, categoria=categoria)

@app.route('/reflexion/<int:id>')
def mostrar_reflexion(id):
    reflexion = Reflexion.query.get_or_404(id)
    notification_body = {
        "contents": {"en": f"Nueva reflexión visitada: {reflexion.titulo}"},
        "included_segments": ["Subscribed Users"]
    }
    try:
        response = onesignal_client.notification_create(notification_body)
        print(f"Notificación enviada: {response.body}")
    except Exception as e:
        print(f"Error enviando notificación: {e}")
    return render_template('reflexion.html', reflexion=reflexion)

@app.route('/articulo-aleatorio')
def articulo_aleatorio():
    reflexion = random.choice(Reflexion.query.all())
    return redirect(url_for('mostrar_reflexion', id=reflexion.id))

@app.route('/galeria', defaults={'page': 1})
@app.route('/galeria/page/<int:page>')
@cache.cached(timeout=300, query_string=True)
def galeria(page):
    per_page = 20
    reflexiones = Reflexion.query.paginate(page=page, per_page=per_page, error_out=False)
    print(f"Galería - Página {page}: {len(reflexiones.items)} reflexiones enviadas de {reflexiones.total} totales")
    return render_template('galeria.html', reflexiones=reflexiones.items, pagination=reflexiones)

@app.route('/recursos')
def recursos():
    return render_template('recursos.html')

@app.route('/contacto', methods=['GET', 'POST'])
def contacto():
    if request.method == 'POST':
        nombre = request.form['nombre']
        email = request.form['email']
        mensaje = request.form['mensaje']
        msg = Message(
            subject=f"Nuevo mensaje de {nombre}",
            recipients=[app.config['MAIL_USERNAME']],
            body=f"Nombre: {nombre}\nCorreo: {email}\nMensaje: {mensaje}"
        )
        mail.send(msg)
        flash('¡Mensaje enviado con éxito! Gracias por contactarnos.', 'success')
        return redirect(url_for('contacto'))
    return render_template('contacto.html')

@app.route('/buscar', methods=['GET'])
@cache.cached(timeout=60, query_string=True)
def buscar():
    query = request.args.get('q', '').strip().lower()
    page = request.args.get('page', 1, type=int)
    per_page = 10
    if not query:
        return render_template('busqueda.html', resultados=[], query=query, page=page, per_page=per_page, total=0)
    resultados = Reflexion.query.filter(
        db.or_(Reflexion.titulo.ilike(f'%{query}%'), Reflexion.contenido.ilike(f'%{query}%'))
    ).all()
    total = len(resultados)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_resultados = resultados[start:end]
    return render_template('busqueda.html', resultados=paginated_resultados, query=query, page=page, per_page=per_page, total=total)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            flash('¡Inicio de sesión exitoso!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Correo o contraseña incorrectos.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(email=email).first():
            flash('El correo ya está registrado.', 'error')
        else:
            user = User(email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('¡Registro exitoso! Por favor, inicia sesión.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('¡Has cerrado sesión!', 'success')
    return redirect(url_for('home'))

@app.route('/toggle_favorite/<int:reflexion_id>', methods=['POST'])
@login_required
def toggle_favorite(reflexion_id):
    reflexion = Reflexion.query.get_or_404(reflexion_id)
    if reflexion in current_user.favorite_reflexiones:
        current_user.favorite_reflexiones.remove(reflexion)
        flash('Reflexión eliminada de favoritos.', 'success')
    else:
        current_user.favorite_reflexiones.append(reflexion)
        flash('Reflexión añadida a favoritos.', 'success')
    db.session.commit()
    return redirect(request.referrer or url_for('mostrar_reflexion', id=reflexion_id))

@app.route('/favoritos')
@login_required
def favoritos():
    return render_template('favoritos.html', favoritos=current_user.favorite_reflexiones)

@app.route('/download_pdf/<int:reflexion_id>')
def download_pdf(reflexion_id):
    reflexion = Reflexion.query.get_or_404(reflexion_id)
    html = render_template('reflexion_pdf.html', 
                          reflexion=reflexion, 
                          año_actual=datetime.datetime.now().year)
    pdf_file = io.BytesIO()
    HTML(string=html).write_pdf(pdf_file)
    pdf_file.seek(0)
    response = make_response(pdf_file.read())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename="{reflexion.titulo}.pdf"'
    return response

# Ruta para ConsciencIA con Gemini
@app.route('/consciencia', methods=['GET', 'POST'])
def mostrar_consciencia():
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    if request.method == 'POST':
        message = request.json.get('message', '').strip()
        session['chat_history'].append(f"Usuario: {message}")
        history = "\n".join(session['chat_history'][-3:])
        
        context = (
            "Eres ConsciencIA, una IA empática y experta en autoayuda basada en 'Voy Consciente'. "
            "Habla en español con un tono serio, cálido y profesional, como un guía confiable. "
            "Usa el historial de la conversación para mantener la coherencia y responder de manera relevante al mensaje actual. "
            "Si el usuario saluda (ej. 'Hola'), responde breve y amablemente (máximo 20 palabras). "
            "Si expresa emociones o preguntas, da respuestas profundas, coherentes y útiles, sin repetir saludos innecesarios ni frases genéricas. "
            f"Historial: {history}"
        )
        try:
            print(f"Enviando solicitud a Gemini con contexto de {len(context)} caracteres")
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                f"{context}\nUsuario: {message}",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=50 if "hola" in message.lower() else 500,
                    temperature=0.6
                )
            )
            session['chat_history'].append(f"ConsciencIA: {response.text}")
            print(f"Respuesta recibida: {response.text}")
            return jsonify({'response': response.text.strip()})
        except Exception as e:
            error_msg = f"Error con Gemini API: {str(e)}"
            print(error_msg)
            return jsonify({'response': "Lo siento, hubo un problema técnico. ¿Intentamos de nuevo?"})
    
    session['chat_history'] = []
    return render_template('consciencia.html')

# Rutas para el editor de reflexiones
@app.route('/editor')
@login_required
def editor():
    reflexiones = Reflexion.query.all()
    return render_template('editor.html', reflexiones=reflexiones)

@app.route('/editor/<int:id>', methods=['GET', 'POST'])
@login_required
def editar_reflexion(id):
    reflexion = Reflexion.query.get_or_404(id)
    if request.method == 'POST':
        reflexion.titulo = request.form['titulo']
        reflexion.contenido = request.form['contenido']
        reflexion.categoria = request.form['categoria']
        reflexion.imagen = request.form['imagen']
        reflexion.fecha = request.form['fecha']
        try:
            db.session.commit()
            flash('Reflexión actualizada correctamente.', 'success')
            return render_template('editar_reflexion.html', reflexion=reflexion)
        except Exception as e:
            db.session.rollback()
            flash(f'Error al actualizar: {str(e)}', 'error')
            return render_template('editar_reflexion.html', reflexion=reflexion)
    return render_template('editar_reflexion.html', reflexion=reflexion)

def schedule_reflections():
    with app.app_context():
        for frequency in ['daily', 'weekly', 'monthly']:
            subscribers = Subscriber.query.filter_by(frequency=frequency).all()
            if subscribers:
                if frequency == 'daily':
                    scheduler.add_job(func=send_weekly_reflection, trigger='interval', days=1, id=f'reflection_{frequency}')
                elif frequency == 'weekly':
                    scheduler.add_job(func=send_weekly_reflection, trigger='interval', weeks=1, id=f'reflection_{frequency}')
                elif frequency == 'monthly':
                    scheduler.add_job(func=send_weekly_reflection, trigger='interval', weeks=4, id=f'reflection_{frequency}')

if __name__ == '__main__':
    with app.app_context():
        print("Tablas gestionadas por Alembic")
        # Migración de subscribers.db (si aún es necesario)
        if os.path.exists('/Users/sebastianredigonda/Desktop/voy_consciente/subscribers.db'):
            old_conn = sqlite3.connect('/Users/sebastianredigonda/Desktop/voy_consciente/subscribers.db')
            old_cursor = old_conn.execute('SELECT email, subscription_date FROM subscribers')
            subscribers = old_cursor.fetchall()
            old_conn.close()
            for email, subscription_date in subscribers:
                if not Subscriber.query.filter_by(email=email).first():
                    subscriber = Subscriber(email=email, subscription_date=subscription_date)
                    db.session.add(subscriber)
            db.session.commit()
            print("Suscriptores migrados desde subscribers.db a basededatos.db")

    scheduler = BackgroundScheduler()
    schedule_reflections()
    scheduler.start()
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True  # Sin ssl_context
    )

   # scheduler.add_job(
   # func=send_weekly_reflection,
   # trigger='cron',
   # day_of_week='mon',           # Lunes
   # hour=9,                      # 9:00 AM
   # minute=0
#)