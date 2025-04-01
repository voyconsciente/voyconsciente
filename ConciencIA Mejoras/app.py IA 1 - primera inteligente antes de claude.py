import json
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, make_response, session
from flask_mail import Mail, Message
from flask_migrate import Migrate
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
from datetime import timedelta, UTC
import random
import os
import sqlite3
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.oauth2 import service_account
import stripe
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from urllib.parse import urlparse, urljoin
from sqlalchemy.sql import text
import requests
import pdfplumber
import random
import datetime
import re

# Crear la aplicación Flask primero
app = Flask(__name__)

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de la sesión permanente
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
app.config['SESSION_COOKIE_SECURE'] = True  # Solo HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Previene acceso desde JavaScript
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Protección CSRF
app.config['SESSION_COOKIE_NAME'] = 'voy_session'  # Nombre único para la cookie

# Configuración de la clave secreta
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    raise ValueError("Error: No se encontró SECRET_KEY en las variables de entorno.")

# Depuración: Verificar qué variables se cargaron
print("Variables de entorno cargadas:")
print(f"STRIPE_SECRET_KEY: {os.getenv('STRIPE_SECRET_KEY')}")
print(f"SECRET_KEY: {os.getenv('SECRET_KEY')}")
print(f"GROK_API_KEY: {os.getenv('GROK_API_KEY')}")

# Configuración de Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Definir la función is_safe_url globalmente
def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    dest_url = urlparse(urljoin(request.host_url, target))
    return dest_url.scheme in ('http', 'https') and ref_url.netloc == dest_url.netloc

# Inicializar el serializador para generar tokens seguros
serializer = URLSafeTimedSerializer(app.secret_key)

# Configuración de Google Analytics
GA_CREDENTIALS_PATH = os.getenv('GA_CREDENTIALS_PATH', os.path.join(app.root_path, 'voy-consciente-analytics.json'))
GA_PROPERTY_ID = os.getenv('GA_PROPERTY_ID', '480922494')
GA_FLOW_ID = os.getenv('GA_FLOW_ID', '10343079148')
GA_FLOW_NAME = os.getenv('GA_FLOW_NAME', 'Voy Consciente')
GA_FLOW_URL = os.getenv('GA_FLOW_URL', 'https://192.168.0.213:5001')

credentials = service_account.Credentials.from_service_account_file(GA_CREDENTIALS_PATH)
analytics_client = BetaAnalyticsDataClient(credentials=credentials)

# Configuración de la API de Grok
grok_api_key = os.getenv("GROK_API_KEY")
if not grok_api_key:
    raise ValueError("Error: No se encontró la clave API de Grok en las variables de entorno.")
else:
    print(f"Clave API de Grok cargada: {grok_api_key[:5]}...")

# Configuración de Stripe
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
if not STRIPE_SECRET_KEY:
    raise ValueError("Error: No se encontró STRIPE_SECRET_KEY en las variables de entorno.")
else:
    print(f"Clave secreta de Stripe cargada: {STRIPE_SECRET_KEY[:5]}...")
stripe.api_key = STRIPE_SECRET_KEY

# Configuración de la base de datos
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///Users/sebastianredigonda/Desktop/voy_consciente/basededatos.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Definición de modelos
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_premium = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=False)
    activation_token = db.Column(db.String(100), nullable=True)
    subscription_date = db.Column(db.String(30), nullable=True)
    preferred_categories = db.Column(db.String(200), default='all')
    frequency = db.Column(db.String(20), default='weekly')
    push_enabled = db.Column(db.Boolean, default=False)
    onesignal_player_id = db.Column(db.String(50))
    birth_date = db.Column(db.String(30), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_activation_token(self):
        serializer = URLSafeTimedSerializer(app.secret_key)
        self.activation_token = serializer.dumps(self.email, salt='activation-salt')
        return self.activation_token

class Reflexion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    titulo = db.Column(db.String(200), nullable=False)
    contenido = db.Column(db.Text, nullable=False)
    fecha = db.Column(db.String(10))
    categoria = db.Column(db.String(50))
    imagen = db.Column(db.String(200))

class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    session_id = db.Column(db.String(100), nullable=True)
    interaction_date = db.Column(db.Date, nullable=False)
    interaction_count = db.Column(db.Integer, default=0)

    @staticmethod
    def get_interaction_count(identifier, is_authenticated):
        today = datetime.datetime.now(UTC).date()
        if is_authenticated:
            interaction = Interaction.query.filter_by(user_id=identifier, interaction_date=today).first()
        else:
            interaction = Interaction.query.filter_by(session_id=identifier, interaction_date=today).first()

        if not interaction:
            interaction = Interaction(
                user_id=identifier if is_authenticated else None,
                session_id=identifier if not is_authenticated else None,
                interaction_date=today,
                interaction_count=0
            )
            db.session.add(interaction)
            db.session.commit()

        return interaction.interaction_count

    @staticmethod
    def increment_interaction(identifier, is_authenticated):
        today = datetime.datetime.now(UTC).date()
        if is_authenticated:
            interaction = Interaction.query.filter_by(user_id=identifier, interaction_date=today).first()
        else:
            interaction = Interaction.query.filter_by(session_id=identifier, interaction_date=today).first()

        if not interaction:
            interaction = Interaction(
                user_id=identifier if is_authenticated else None,
                session_id=identifier if not is_authenticated else None,
                interaction_date=today,
                interaction_count=0
            )
        interaction.interaction_count += 1
        db.session.add(interaction)
        db.session.commit()
        return interaction.interaction_count

class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    file_name = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f"<Book {self.title}>"

favorite = db.Table('favorite',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('reflexion_id', db.Integer, db.ForeignKey('reflexion.id'), primary_key=True)
)

User.favorite_reflexiones = db.relationship('Reflexion', secondary=favorite, backref=db.backref('favorited_by', lazy='dynamic'))

class PasswordReset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), nullable=False)
    token = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Depuración dentro del contexto de la aplicación
with app.app_context():
    inspector = inspect(db.engine)
    print("Tablas en la base de datos:", inspector.get_table_names())
    print("Modelo User definido:", hasattr(User, '__table__'))

# Configuración de caché
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configuración de Flask-Mail
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

# Configuración de OneSignal
onesignal_client = OneSignalClient(
    app_id=os.getenv("ONESIGNAL_APP_ID"),
    rest_api_key=os.getenv("ONESIGNAL_REST_API_KEY")
)
if not onesignal_client.app_id or not onesignal_client.rest_api_key:
    raise ValueError("Error: Faltan ONESIGNAL_APP_ID o ONESIGNAL_REST_API_KEY en las variables de entorno.")

# Función para cargar PDFs en la base de datos
PDF_FOLDER = os.path.join(app.static_folder, 'books')

def load_pdfs_to_db():
    with app.app_context():
        AVAILABLE_BOOKS = [
            {"title": "El hombre en busca de sentido", "file": "El hombre en busca de sentido - Viktor Frankl.pdf"},
            {"title": "Cómo hacer que te pasen cosas buenas", "file": "Como hacer que te pasen cosas buenas - Marian Rojas Estape.pdf"},
            {"title": "Los secretos de la mente millonaria", "file": "Los secretos de la mente millonaria - T. Harv Eker.pdf"},
            {"title": "Piense y hágase rico", "file": "Piense y hagase rico - Napoleon Hill.pdf"},
            {"title": "La inteligencia emocional", "file": "La inteligencia emocional - Daniel Goleman.pdf"},
            {"title": "Los 7 hábitos de la gente altamente efectiva", "file": "Los 7 habitos de la gente altamente efec - Stephen R. Covey.pdf"},
            {"title": "Cómo ganar amigos e influir sobre las personas", "file": "Como ganar amigos e influir sobre las pe - Dale Carnegie.pdf"},
            {"title": "Tus zonas erróneas", "file": "Tus zonas erroneas - Wayne W. Dyer.pdf"},
            {"title": "Tus zonas mágicas", "file": "Tus zonas magicas - Wayne W Dyer.pdf"},
            {"title": "12 reglas para vivir", "file": "12 reglas para vivir - Jordan Peterson.pdf"},
            {"title": "El poder de los hábitos", "file": "El poder de los habitos - Charles Duhigg.pdf"},
            {"title": "Hábitos atómicos", "file": "Habitos atomicos - James Clear.pdf"},
            {"title": "La magia del orden", "file": "La magia del orden - Marie Kondo.pdf"},
            {"title": "El monje que vendió su Ferrari", "file": "El monje que vendio su Ferrari - Robin S. Sharma.pdf"},
            {"title": "El poder del ahora", "file": "El poder del ahora - Eckhart Tolle.pdf"}
        ]
        
        for book in AVAILABLE_BOOKS:
            pdf_path = os.path.join(PDF_FOLDER, book["file"])
            existing_book = Book.query.filter_by(file_name=book["file"]).first()
            if not existing_book and os.path.exists(pdf_path):
                with pdfplumber.open(pdf_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"
                    if full_text:
                        new_book = Book(
                            title=book["title"],
                            file_name=book["file"],
                            content=full_text
                        )
                        db.session.add(new_book)
                        print(f"Cargado '{book['title']}' en la base de datos.")
                db.session.commit()
            elif existing_book:
                print(f"'{book['title']}' ya está en la base de datos.")
            else:
                print(f"No se encontró '{pdf_path}'.")

# Función para enviar reflexiones programadas
def send_weekly_reflection():
    with app.app_context():
        users = User.query.filter_by(is_active=True).all()
        for user in users:
            categories = user.preferred_categories.split(',')
            if 'all' in categories:
                reflection = Reflexion.query.order_by(db.func.random()).first()
            else:
                reflection = Reflexion.query.filter(Reflexion.categoria.in_(categories)).order_by(db.func.random()).first()

            if not reflection:
                print(f"No hay reflexiones para {user.email} en {categories}")
                continue

            msg = Message('Tu Reflexión Personalizada - Voy Consciente', recipients=[user.email])
            msg.body = f'Reflexión: "{reflection.contenido}"'
            msg.html = f"""
            <html>
                <body style="font-family: Arial, sans-serif; color: #333; background-color: #f9f9f9; padding: 20px;">
                    <div style="max-width: 600px; margin: 0 auto; background-color: #fff; border-radius: 10px; padding: 20px;">
                        <h1 style="color: #4CAF50; text-align: center;">Tu Reflexión Personalizada</h1>
                        <p style="font-style: italic;">"{reflection.contenido}"</p>
                        <p style="text-align: center;">- {reflection.titulo}</p>
                        <p style="text-align: center;"><a href="{GA_FLOW_URL}/reflexion/{reflection.id}" style="color: #4CAF50;">Leer más</a></p>
                    </div>
                </body>
            </html>
            """
            try:
                mail.send(msg)
                print(f'Correo enviado a {user.email}')
            except Exception as e:
                print(f'Error enviando a {user.email}: {str(e)}')

            if user.push_enabled and user.onesignal_player_id:
                notification = {
                    "contents": {"en": f"{reflection.titulo}: {reflection.contenido[:50]}..."},
                    "include_player_ids": [user.onesignal_player_id]
                }
                try:
                    response = onesignal_client.notification_create(notification)
                    print(f'Notificación push enviada a {user.email}: {response.body}')
                except Exception as e:
                    print(f'Error enviando push a {user.email}: {str(e)}')

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

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Has cerrado sesión exitosamente.', 'success')
    return redirect(url_for('home'))

@app.route('/update_push', methods=['POST'])
def update_push():
    data = request.get_json()
    email = data.get('email')
    player_id = data.get('player_id')
    if email:
        user = User.query.filter_by(email=email).first()
        if user:
            user.onesignal_player_id = player_id
            db.session.commit()
            return jsonify({'status': 'success'})
    return jsonify({'status': 'error'})

@app.route('/terminos-condiciones')
def terminos_condiciones():
    return render_template('terminos_condiciones.html', año_actual=datetime.datetime.now().year)

@app.route('/politica-privacidad')
def politica_privacidad():
    return render_template('politica_privacidad.html', año_actual=datetime.datetime.now().year)

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

@app.route('/consciencia-info')
def consciencia_info():
    return render_template('consciencia_info.html')

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
    if current_user.is_authenticated:
        next_page = request.args.get('next', '')
        print(f"Usuario ya autenticado, redirigiendo a: {next_page if next_page and is_safe_url(next_page) else url_for('home')}")
        if next_page and is_safe_url(next_page):
            return redirect(next_page)
        return redirect(url_for('home'))

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        next_page = request.form.get('next', request.args.get('next', ''))

        if user and user.check_password(password):
            if not user.is_active:
                flash('Tu cuenta aún no está activada. Por favor, revisa tu correo para activar tu cuenta.', 'error')
                return render_template('login.html', next=next_page, email=email)
            session.permanent = True  # Hacer la sesión permanente
            login_user(user, remember=True)  # Habilitar "recordarme" para persistencia
            print(f"Inicio de sesión exitoso para {email}. Redirigiendo a: {next_page if next_page and is_safe_url(next_page) else url_for('home')}")
            if next_page and is_safe_url(next_page):
                return redirect(next_page)
            return redirect(url_for('home'))
        else:
            flash('Correo o contraseña incorrectos.', 'error')
            print(f"Inicio de sesión fallido para {email}")
            return render_template('login.html', next=next_page, email=email)

    next_page = request.args.get('next', '')
    email = request.args.get('email', '')
    print(f"Valor de next_page en GET: {next_page}")
    return render_template('login.html', next=next_page, email=email)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        name = request.form['name']
        birth_date_str = request.form['birth_date']
        phone = request.form.get('phone', '')
        next_page = request.args.get('next', '')

        # Validación de fecha de nacimiento
        try:
            birth_date = datetime.datetime.strptime(birth_date_str, '%d/%m/%Y')
            birth_date_iso = birth_date.isoformat()
        except ValueError:
            flash('El formato de la fecha de nacimiento debe ser DD/MM/AAAA.', 'error')
            return render_template('register.html', next=next_page, name=name, email=email, birth_date=birth_date_str, phone=phone, password=password, confirm_password=confirm_password)

        # Validación de contraseña
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'
        if password != confirm_password:
            flash('Las contraseñas no coinciden.', 'error')
            return render_template('register.html', next=next_page, name=name, email=email, birth_date=birth_date_str, phone=phone, password=password, confirm_password=confirm_password)
        if not re.match(password_pattern, password):
            flash('La contraseña debe tener al menos 8 caracteres, una mayúscula, una minúscula y un número.', 'error')
            return render_template('register.html', next=next_page, name=name, email=email, birth_date=birth_date_str, phone=phone, password=password, confirm_password=confirm_password)

        # Verificar si el correo ya está registrado
        if User.query.filter_by(email=email).first():
            flash('El correo ya está registrado.', 'error')
            return render_template('register.html', next=next_page, name=name, email=email, birth_date=birth_date_str, phone=phone, password=password, confirm_password=confirm_password)

        # Crear usuario
        user = User(
            email=email,
            subscription_date=datetime.datetime.now().isoformat(),
            preferred_categories='all',
            frequency='weekly',
            push_enabled=False,
            birth_date=birth_date_iso
        )
        user.set_password(password)
        user.is_active = False
        token = user.generate_activation_token()
        db.session.add(user)
        db.session.commit()

        # Enviar correo de activación
        activation_link = url_for('activate', token=token, next=next_page, _external=True)
        msg = Message(
            subject='Activa tu cuenta en Voy Consciente',
            recipients=[email],
            body=f'''
            Hola {name},

            Gracias por registrarte en Voy Consciente. Para activar tu cuenta, haz clic en el siguiente enlace:

            {activation_link}

            ¡Estás suscrito para recibir nuestras reflexiones semanales por correo!

            Si no te registraste, puedes ignorar este correo.

            ¡Esperamos verte pronto!
            El equipo de Voy Consciente
            '''
        )
        try:
            mail.send(msg)
            flash('Se ha enviado un correo para activar tu cuenta. Revisa tu bandeja de entrada (y spam).', 'success')
        except Exception as e:
            flash(f'Error al enviar el correo de activación: {str(e)}', 'error')
            db.session.delete(user)
            db.session.commit()
            return render_template('register.html', next=next_page, name=name, email=email, birth_date=birth_date_str, phone=phone, password=password, confirm_password=confirm_password)

        return redirect(url_for('login', next=next_page))

    # Para método GET, renderizamos el formulario sin datos precargados
    next_page = request.args.get('next', '')
    return render_template('register.html', next=next_page)

@app.route('/activate/<token>')
def activate(token):
    try:
        email = serializer.loads(token, salt='activation-salt', max_age=3600)
        user = User.query.filter_by(email=email).first()
        if not user:
            flash('El enlace de activación es inválido o el usuario no existe.', 'error')
            return redirect(url_for('register', next=request.args.get('next')))

        if user.is_active:
            flash('Tu cuenta ya está activada.', 'success')
            next_page = request.args.get('next', '')
            if next_page and is_safe_url(next_page):
                return redirect(next_page)
            return redirect(url_for('home'))

        user.is_active = True
        user.activation_token = None
        db.session.commit()
        flash('¡Tu cuenta ha sido activada! Por favor, inicia sesión.', 'success')
        next_page = request.args.get('next', '')
        return redirect(url_for('login', next=next_page))
    except (SignatureExpired, BadSignature):
        flash('El enlace de activación ha expirado o es inválido.', 'error')
        return redirect(url_for('register', next=request.args.get('next')))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            token = serializer.dumps(email, salt='password-reset-salt')
            reset_link = url_for('reset_password', token=token, _external=True)

            reset_record = PasswordReset(email=email, token=token)
            db.session.add(reset_record)
            db.session.commit()

            msg = Message('Restablecer tu contraseña - Voy Consciente', recipients=[email])
            msg.body = f'Haz clic en el siguiente enlace para restablecer tu contraseña: {reset_link}\nEste enlace expira en 1 hora.'
            msg.html = f"""
            <html>
                <body style="font-family: Arial, sans-serif; color: #333; background-color: #f9f9f9; padding: 20px;">
                    <div style="max-width: 600px; margin: 0 auto; background-color: #fff; border-radius: 10px; padding: 20px;">
                        <h1 style="color: #4CAF50; text-align: center;">Restablecer Contraseña</h1>
                        <p>Haz clic en el enlace para restablecer tu contraseña:</p>
                        <p style="text-align: center;"><a href="{reset_link}" style="color: #4CAF50;">Restablecer Contraseña</a></p>
                        <p>Este enlace expira en 1 hora.</p>
                    </div>
                </body>
            </html>
            """
            try:
                mail.send(msg)
                flash('Se ha enviado un enlace de restablecimiento a tu correo.', 'success')
            except Exception as e:
                flash(f'Error al enviar el correo: {str(e)}', 'error')
        else:
            flash('No se encontró un usuario con ese correo.', 'error')
        return redirect(url_for('forgot_password'))

    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = serializer.loads(token, salt='password-reset-salt', max_age=3600)
        user = User.query.filter_by(email=email).first()
        if not user:
            flash('Token inválido o usuario no encontrado.', 'error')
            return redirect(url_for('login'))

        if request.method == 'POST':
            new_password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')

            if new_password != confirm_password:
                flash('Las contraseñas no coinciden. Por favor, intenta de nuevo.', 'error')
                return render_template('reset_password.html', token=token)

            password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'
            import re
            if not re.match(password_pattern, new_password):
                flash('La contraseña debe tener al menos 8 caracteres, una mayúscula, una minúscula y un número.', 'error')
                return render_template('reset_password.html', token=token)

            user.set_password(new_password)
            PasswordReset.query.filter_by(email=email).delete()
            db.session.commit()
            flash('Tu contraseña ha sido restablecida. Inicia sesión con tu nueva contraseña.', 'success')
            return redirect(url_for('login'))

        return render_template('reset_password.html', token=token)
    except Exception as e:
        flash('El enlace de restablecimiento ha expirado o es inválido.', 'error')
        return redirect(url_for('forgot_password'))

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

import random
import datetime

@app.route('/consciencia', methods=['GET', 'POST'])
@login_required
def mostrar_consciencia():
    # Inicializar variables de sesión con una estructura más rica y semántica
    session_vars = {
        'chat_history': [],              # Histórico completo
        'user_topics': {},               # Temas de interés con ponderación
        'user_preferences': {            # Preferencias estructuradas
            'communication_style': 'auto', # formal, casual, empático, analítico
            'response_depth': 'adaptive',  # conciso, moderado, profundo
            'interests': [],             # Intereses detectados
            'cognitive_preferences': [], # Visual, analítico, emocional, etc.
        },
        'conversation_context': {        # Contexto enriquecido
            'thread_id': str(uuid.uuid4()),
            'current_topic': None,       
            'topic_retention': 0,        # 0-10 cuánto persiste en un tema
            'cognitive_load': 0,         # 0-10 complejidad percibida
            'unanswered_questions': [],  # Preguntas pendientes de respuesta
        },
        'conversation_dynamics': {       # Métricas dinámicas
            'depth': 0,                  # Profundidad actual 
            'engagement': 0,             # Nivel de compromiso 0-10
            'initiative_ratio': 0.5,     # Balance entre usuario/IA
            'topic_continuity': 0,       # Continuidad temática
        },
        'emotional_context': {           # Análisis emocional complejo
            'current_state': 'neutral',
            'intensity': 0,              # 0-10
            'valence': 0,                # -5 (muy negativo) a +5 (muy positivo)
            'arousal': 0,                # 0-10 nivel de activación
            'trend': 'stable',           # increasing, decreasing, fluctuating
            'history': []                # Registro histórico de emociones
        },
        'session_metrics': {             # Métricas ampliadas
            'session_count': 0,
            'message_count': 0,
            'avg_response_time': 0,
            'avg_message_length': 0,
            'interaction_quality': 0,    # 0-10 calidad percibida
        },
        'temporal_context': {            # Contexto temporal
            'user_timezone': None,
            'user_specified_hour': None,
            'session_start_time': time.time(),
            'last_interaction_time': time.time(),
        }
    }
    
    # Inicializar o recuperar estructura de sesión
    for category, values in session_vars.items():
        if category not in session:
            session[category] = values
    
    # Actualizar métricas de sesión (solo en GET)
    if request.method == 'GET':
        session['session_metrics']['session_count'] += 1
        session['session_metrics']['message_count'] = 0
        session['conversation_dynamics']['depth'] = 0
        session['temporal_context']['session_start_time'] = time.time()
        session['conversation_context']['thread_id'] = str(uuid.uuid4())
        
        # Mantener preferencias y emociones para continuidad
        
    identifier = current_user.id
    is_premium = current_user.is_premium
    is_admin = current_user.is_admin

    # Sistema adaptativo de límites de interacción
    FREE_INTERACTION_LIMIT = 5
    PREMIUM_DAILY_LIMIT = 50
    
    interaction_count = Interaction.get_interaction_count(identifier, is_authenticated=True)
    
    # Cálculo de límites con bonificaciones
    user_score = UserScore.get_score(identifier)
    bonus_interactions = min(5, user_score // 100)  # Bonificación por puntuación
    
    effective_limit = FREE_INTERACTION_LIMIT + bonus_interactions
    
    if is_premium:
        effective_limit = PREMIUM_DAILY_LIMIT
    elif is_admin:
        effective_limit = 999999
        
    remaining_interactions = max(0, effective_limit - interaction_count)

    # Sistemas de análisis semántico avanzado
    
    # 1. Análisis de sentimiento con modelo NLP contextual
    def analyze_sentiment(text, history=None):
        """
        Análisis de sentimiento avanzado usando NLP
        Retorna un diccionario completo con diferentes dimensiones emocionales
        """
        try:
            # Versión simplificada - en producción usar un modelo NLP completo
            positive_words = {
                'feliz': 0.8, 'alegre': 0.7, 'gracias': 0.6, 'genial': 0.8, 
                'bueno': 0.5, 'excelente': 0.9, 'contento': 0.7, 'satisfecho': 0.6,
                'optimista': 0.7, 'energía': 0.6, 'emocionado': 0.8
            }
            
            negative_words = {
                'triste': -0.7, 'enojado': -0.8, 'frustrado': -0.7, 'malo': -0.6, 
                'terrible': -0.9, 'problema': -0.5, 'preocupado': -0.6, 
                'ansioso': -0.7, 'molesto': -0.6, 'decepcionado': -0.7
            }
            
            # Palabras de alta intensidad emocional
            high_arousal = {
                'emocionado', 'furioso', 'extasiado', 'aterrorizado', 'eufórico',
                'ansioso', 'pánico', 'urgente', 'crisis'
            }
            
            text_lower = text.lower()
            words = re.findall(r'\w+', text_lower)
            
            # Análisis de valencia (positivo/negativo)
            valence_scores = []
            for word in words:
                if word in positive_words:
                    valence_scores.append(positive_words[word])
                elif word in negative_words:
                    valence_scores.append(negative_words[word])
            
            # Calcular valencia promedio
            valence = sum(valence_scores) / max(1, len(valence_scores)) if valence_scores else 0
            
            # Intensidad emocional (presencia de palabras intensas o signos de exclamación)
            intensity = min(10, (sum(1 for word in words if word in high_arousal) * 2) + 
                          (text.count('!') * 1.5) + 
                          (sum(1 for c in text if c.isupper()) / max(1, len(text)) * 10))
            
            # Determinar estado emocional
            if valence > 0.3:
                state = 'positive'
            elif valence < -0.3:
                state = 'negative'
            else:
                state = 'neutral'
                
            # Añadir matices
            if state == 'positive' and intensity > 7:
                state = 'excited'
            elif state == 'negative' and intensity > 7:
                state = 'distressed'
            
            # Nivel de activación (arousal)
            arousal = min(10, intensity + abs(valence) * 5)
            
            # Análisis de tendencia con historia previa
            trend = 'stable'
            if history and len(history) > 2:
                prev_valence = history[-1]['valence']
                if valence > prev_valence + 0.3:
                    trend = 'improving'
                elif valence < prev_valence - 0.3:
                    trend = 'deteriorating'
            
            return {
                'state': state,
                'valence': round(valence * 5, 1),  # Escala -5 a +5
                'intensity': round(intensity, 1),
                'arousal': round(arousal, 1),
                'trend': trend
            }
                
        except Exception as e:
            print(f"Error en análisis de sentimiento: {e}")
            return {'state': 'neutral', 'valence': 0, 'intensity': 0, 'arousal': 0, 'trend': 'stable'}

    # 2. Extracción avanzada de temas con categorización jerárquica
    def extract_topics(text, history=None):
        """
        Extracción avanzada de temas utilizando análisis semántico
        Retorna diccionario con temas primarios y secundarios con ponderación
        """
        # Taxonomía jerárquica de temas
        topic_taxonomy = {
            'desarrollo_personal': {
                'keywords': ['crecimiento', 'desarrollo', 'mejora', 'hábitos', 'metas'],
                'subtopics': {
                    'productividad': ['productividad', 'eficiencia', 'organización', 'gestión', 'tiempo'],
                    'meditación': ['meditación', 'mindfulness', 'atención', 'presente', 'calma'],
                    'propósito': ['propósito', 'significado', 'misión', 'valores', 'objetivos']
                }
            },
            'filosofía': {
                'keywords': ['filosofía', 'existencia', 'significado', 'propósito', 'ética'],
                'subtopics': {
                    'estoicismo': ['estoicismo', 'marcus', 'aurelio', 'seneca', 'epicteto'],
                    'existencialismo': ['existencialismo', 'sartre', 'existencia', 'libertad', 'absurdo'],
                    'metafísica': ['metafísica', 'realidad', 'ser', 'existencia', 'ontología']
                }
            },
            'psicología': {
                'keywords': ['psicología', 'mente', 'conducta', 'comportamiento', 'cerebro'],
                'subtopics': {
                    'emociones': ['emoción', 'sentimiento', 'afecto', 'gestión emocional'],
                    'cognitivo': ['cognitivo', 'pensamiento', 'creencia', 'sesgo', 'mental'],
                    'personalidad': ['personalidad', 'carácter', 'temperamento', 'rasgo']
                }
            },
            'relaciones': {
                'keywords': ['relación', 'amistad', 'pareja', 'interpersonal', 'social'],
                'subtopics': {
                    'comunicación': ['comunicación', 'diálogo', 'conversar', 'escucha'],
                    'conflictos': ['conflicto', 'desacuerdo', 'pelea', 'resolución'],
                    'intimidad': ['intimidad', 'conexión', 'vulnerabilidad', 'confianza']
                }
            },
            'trabajo': {
                'keywords': ['trabajo', 'empleo', 'carrera', 'profesión', 'laboral'],
                'subtopics': {
                    'liderazgo': ['liderazgo', 'líder', 'dirigir', 'equipo', 'influencia'],
                    'satisfacción': ['satisfacción', 'realización', 'propósito', 'motivación'],
                    'estrés': ['estrés', 'burnout', 'presión', 'ansiedad', 'agotamiento']
                }
            },
            'salud': {
                'keywords': ['salud', 'bienestar', 'enfermedad', 'médico', 'cuerpo'],
                'subtopics': {
                    'mental': ['mental', 'depresión', 'ansiedad', 'terapia', 'psicológico'],
                    'física': ['física', 'ejercicio', 'nutrición', 'descanso', 'energía'],
                    'hábitos': ['hábito', 'rutina', 'disciplina', 'consistencia', 'cambio']
                }
            },
            'tecnología': {
                'keywords': ['tecnología', 'digital', 'app', 'software', 'computadora'],
                'subtopics': {
                    'ia': ['ia', 'inteligencia artificial', 'machine learning', 'algoritmo'],
                    'internet': ['internet', 'web', 'online', 'digital', 'virtual'],
                    'programación': ['programación', 'código', 'desarrollo', 'software']
                }
            }
        }
        
        try:
            text_lower = text.lower()
            words = set(re.findall(r'\w+', text_lower))
            
            # Preprocesamiento: Eliminar stopwords
            stopwords = {'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'a', 'de', 'en', 'que', 'por', 'con'}
            words = words - stopwords
            
            # Puntuar temas primarios
            primary_scores = {}
            subtopic_scores = {}
            
            for topic, data in topic_taxonomy.items():
                # Puntuar tema principal
                topic_score = sum(2 for keyword in data['keywords'] if keyword in text_lower or 
                                  any(keyword in word for word in words))
                
                if topic_score > 0:
                    primary_scores[topic] = topic_score
                    
                    # Puntuar subtemas
                    for subtopic, keywords in data['subtopics'].items():
                        subtopic_score = sum(1 for keyword in keywords if keyword in text_lower or
                                           any(keyword in word for word in words))
                        
                        if subtopic_score > 0:
                            full_topic = f"{topic}.{subtopic}"
                            subtopic_scores[full_topic] = subtopic_score
            
            # Combinar resultados en formato jerárquico
            results = {
                'primary': {topic: score for topic, score in primary_scores.items() if score > 0},
                'secondary': {topic: score for topic, score in subtopic_scores.items() if score > 0}
            }
            
            # Si hay historia, aumentar score de temas previos para continuidad
            if history and isinstance(history, list) and history:
                for prev_topic in history:
                    if prev_topic in results['primary']:
                        results['primary'][prev_topic] += 1
                    elif prev_topic in results['secondary']:
                        results['secondary'][prev_topic] += 0.5
            
            # Si no se detecta ningún tema, usar un valor por defecto
            if not results['primary'] and not results['secondary']:
                results['primary']['general'] = 1
                
            return results
                
        except Exception as e:
            print(f"Error en extracción de temas: {e}")
            return {'primary': {'general': 1}, 'secondary': {}}

    # 3. Detector de intención conversacional
    def detect_intent(text, context=None):
        """
        Detecta la intención principal del usuario en su mensaje
        """
        try:
            text_lower = text.lower()
            
            # Patrones de intención con expresiones regulares y ponderación
            intent_patterns = {
                'pregunta_abierta': (r'\b(qué|cuál|cómo|por qué|para qué|dónde|cuándo|quién).*\?', 3),
                'pregunta_cerrada': (r'.*\b(es|son|está|están|puede|pueden|ha|han).*\?', 2),
                'solicitud_info': (r'\b(dime|cuéntame|explícame|necesito saber|quiero aprender)\b', 3),
                'solicitud_consejo': (r'\b(aconséjame|deberíamos|debería|qué (harías|hago|me recomiendas))\b', 4),
                'expresión_emocional': (r'\b(me siento|estoy (triste|feliz|preocupado|ansioso|emocionado))\b', 4),
                'reflexión': (r'\b(reflexión|pensar|contemplar|filosofar|meditar|considerar)\b', 3),
                'agradecimiento': (r'\b(gracias|agradezco|te lo agradezco)\b', 3),
                'despedida': (r'\b(adiós|hasta luego|nos vemos|chao|hasta pronto)\b', 4),
                'saludo': (r'\b(hola|buenos días|buenas tardes|buenas noches|saludos)\b', 4),
                'desacuerdo': (r'\b(no estoy de acuerdo|difiero|creo que no|incorrecto|equivocado)\b', 3),
                'acuerdo': (r'\b(estoy de acuerdo|coincido|exacto|así es|correcto)\b', 3),
                'clarificación': (r'\b(no entiendo|podrías aclarar|qué quieres decir|a qué te refieres)\b', 3),
                'preferencia': (r'\b(prefiero|me gusta más|mejor|en lugar de)\b', 2),
                'meta_conversación': (r'\b(hablemos de|cambiemos de tema|me gustaría discutir)\b', 3),
                'configuración': (r'\b(habla más (formal|profesional|casual|informal|cercano))\b', 4),
            }
            
            # Detectar intenciones por patrones
            found_intents = {}
            for intent, (pattern, weight) in intent_patterns.items():
                matches = re.findall(pattern, text_lower)
                if matches:
                    found_intents[intent] = weight * len(matches)
            
            # Añadir detección contextual
            if '?' in text:
                found_intents['pregunta'] = found_intents.get('pregunta', 0) + 2
                
            if '!' in text:
                found_intents['exclamación'] = 2
                
            # Usar contexto previo para mejorar detección
            if context and 'previous_intent' in context:
                prev_intent = context['previous_intent']
                # Dar continuidad a intenciones secuenciales
                if prev_intent in ['pregunta_abierta', 'solicitud_info'] and len(text.split()) < 5:
                    found_intents['continuación'] = 3
            
            # Determinar intención principal
            if found_intents:
                primary_intent = max(found_intents.items(), key=lambda x: x[1])
                secondary_intents = sorted([(i, s) for i, s in found_intents.items() if i != primary_intent[0]], 
                                          key=lambda x: x[1], reverse=True)[:2]
                
                return {
                    'primary': primary_intent[0],
                    'secondary': [i[0] for i in secondary_intents],
                    'confidence': min(10, primary_intent[1])
                }
            else:
                # Si no se detecta intención específica
                return {
                    'primary': 'declarativa',
                    'secondary': [],
                    'confidence': 1
                }
                
        except Exception as e:
            print(f"Error en detección de intención: {e}")
            return {'primary': 'indeterminada', 'secondary': [], 'confidence': 0}

    # 4. Análisis de estilo cognitivo del usuario
    def analyze_cognitive_style(text, history=None):
        """
        Analiza el estilo cognitivo y comunicativo del usuario para adaptar respuestas
        """
        try:
            text_lower = text.lower()
            
            # Indicadores de diferentes estilos cognitivos
            style_indicators = {
                'analítico': [
                    r'\b(analizar|análisis|lógico|datos|evidencia|estadística|precisión|exactitud)\b',
                    r'\b(por (un|otro) lado|sin embargo|no obstante|en contraste)\b'
                ],
                'emocional': [
                    r'\b(siento|sentir|emoción|me afecta|me duele|me alegra|me entristece)\b',
                    r'\b(realmente|sinceramente|honestamente|de corazón)\b'
                ],
                'pragmático': [
                    r'\b(funciona|utilidad|práctico|aplicación|implementar|solución|resultado)\b',
                    r'\b(cómo hago|cómo puedo|pasos para|método)\b'
                ],
                'conceptual': [
                    r'\b(concepto|idea|teoría|filosofía|significado|sentido|esencia)\b',
                    r'\b(¿qué significa|¿por qué|fundamentalmente|en esencia)\b'
                ],
                'visual': [
                    r'\b(ver|imagen|visualizar|imaginar|escenario|panorama|perspectiva)\b',
                    r'\b(mostrar|ilustrar|dibujar|representar)\b'
                ],
                'verbal': [
                    r'\b(explicar|describir|narrar|contar|relatar|expresar)\b',
                    r'\b(en palabras|forma de decir|manera de expresar)\b'
                ],
                'detallista': [
                    r'\b(detalle|específico|precisamente|exactamente|concretamente)\b',
                    r'\b(paso a paso|punto por punto|uno por uno)\b'
                ],
                'holístico': [
                    r'\b(general|panorama|completo|todo|entero|conjunto|sistema)\b',
                    r'\b(en general|como un todo|visión global|gran esquema)\b'
                ]
            }
            
            # Detectar presencia de indicadores
            style_scores = {style: 0 for style in style_indicators}
            for style, patterns in style_indicators.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text_lower)
                    style_scores[style] += len(matches)
            
            # Analizar complejidad lingüística (indicador de estilo cognitivo)
            words = text.split()
            avg_word_length = sum(len(word) for word in words) / max(1, len(words))
            
            complexity = 0
            if len(words) > 30:
                complexity += 1
            if avg_word_length > 6:
                complexity += 1
            if len(re.findall(r'[,;:]', text)) > 3:
                complexity += 1
            if len(re.findall(r'\b(sin embargo|no obstante|por consiguiente|en consecuencia)\b', text_lower)) > 0:
                complexity += 2
                
            # Determinar estilos dominantes
            dominant_styles = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)
            
            result = {
                'primary_style': dominant_styles[0][0] if dominant_styles[0][1] > 0 else 'neutral',
                'secondary_style': dominant_styles[1][0] if len(dominant_styles) > 1 and dominant_styles[1][1] > 0 else None,
                'complexity': complexity,
                'detail_orientation': style_scores['detallista'] - style_scores['holístico'],
                'abstraction_level': style_scores['conceptual'] - style_scores['pragmático'],
                'emotional_analytical_balance': style_scores['emocional'] - style_scores['analítico']
            }
            
            # Incorporar historia si está disponible
            if history and isinstance(history, list) and len(history) > 0:
                # Suavizar cambios para evitar oscilaciones bruscas
                prev_primary = history[-1].get('primary_style')
                if prev_primary and prev_primary != result['primary_style'] and style_scores[prev_primary] > 0:
                    # Dar algo de inercia al estilo previo
                    result['secondary_style'] = result['primary_style']
                    result['primary_style'] = prev_primary
            
            return result
            
        except Exception as e:
            print(f"Error en análisis de estilo cognitivo: {e}")
            return {
                'primary_style': 'neutral',
                'secondary_style': None,
                'complexity': 1,
                'detail_orientation': 0,
                'abstraction_level': 0,
                'emotional_analytical_balance': 0
            }

    # 5. Análisis de contexto y coherencia conversacional
    def analyze_conversation_dynamics(user_message, history, context):
        """
        Analiza la dinámica, coherencia y progresión de la conversación
        """
        try:
            # Extraer métricas básicas
            current_time = time.time()
            time_since_last = current_time - context.get('last_interaction_time', current_time)
            is_new_session = time_since_last > 30 * 60  # 30 minutos
            
            # Analizar longitud de mensaje
            msg_length = len(user_message.split())
            msg_complexity = len(re.findall(r'[.;:]', user_message))
            
            # Evaluar continuidad temática
            topics = extract_topics(user_message, context.get('last_topics', []))
            primary_topics = list(topics['primary'].keys())
            
            topic_continuity = 0
            topic_shift = True
            
            if context.get('current_topic') and context['current_topic'] in primary_topics:
                topic_continuity = min(10, context.get('topic_retention', 0) + 2)
                topic_shift = False
            elif context.get('last_topics') and any(t in primary_topics for t in context['last_topics']):
                topic_continuity = min(10, context.get('topic_retention', 0) + 1)
                topic_shift = False
            else:
                topic_continuity = max(0, context.get('topic_retention', 0) - 3)
                topic_shift = True
            
            # Medir profundidad conversacional
            current_depth = context.get('depth', 0)
            
            # Aumenta con: longitud de mensaje, complejidad, continuidad temática
            depth_change = 0
            
            if msg_length > 30:  # Mensaje largo
                depth_change += 1
            if msg_complexity > 2:  # Mensaje complejo
                depth_change += 1
            if topic_continuity > 5:  # Alta continuidad temática
                depth_change += 1
            if context.get('initiative_ratio', 0.5) < 0.4:  # Usuario toma más iniciativa
                depth_change += 1
                
            new_depth = min(10, current_depth + depth_change)
            if topic_shift:
                new_depth = max(1, new_depth - 1)  # Reducir profundidad en cambios de tema
                
            # Medir nivel de compromiso (engagement)
            engagement = min(10, (
                (msg_length / 20) +        # Longitud normalizada (0-2.5 aprox)
                (msg_complexity) +         # Complejidad (0-5 aprox)
                (depth_change * 2) +       # Cambio en profundidad (0-4)
                (0 if is_new_session else 2)  # Continuidad de sesión
            ))
            
            # Determinar quién está tomando la iniciativa
            # < 0.5 significa que el usuario está dirigiendo más
            # > 0.5 significa que la IA está dirigiendo más
            
            # Calcular basado en preguntas, solicitudes, etc.
            user_intent = detect_intent(user_message, {'previous_intent': context.get('last_intent')})
            
            initiative_markers = {
                'pregunta_abierta': -0.1,    # Usuario toma iniciativa con preguntas
                'solicitud_info': -0.1,      # Usuario solicita información
                'meta_conversación': -0.2,   # Usuario dirige explícitamente la conversación
                'continuación': 0.1,         # Respuesta a iniciativa previa de la IA
            }
            
            # Ajustar ratio de iniciativa basado en intención detectada
            initiative_shift = initiative_markers.get(user_intent['primary'], 0)
            for intent in user_intent['secondary']:
                initiative_shift += initiative_markers.get(intent, 0) * 0.5
                
            # Suavizar cambios para evitar oscilaciones bruscas
            new_initiative = min(1.0, max(0.0, context.get('initiative_ratio', 0.5) + initiative_shift))
            
            return {
                'depth': new_depth,
                'engagement': round(engagement, 1),
                'initiative_ratio': round(new_initiative, 2),
                'topic_continuity': topic_continuity,
                'topic_shift': topic_shift,
                'current_topic': primary_topics[0] if primary_topics else context.get('current_topic'),
                'last_topics': primary_topics[:3],
                'topic_retention': topic_continuity,
                'last_intent': user_intent['primary'],
                'message_length': msg_length,
                'message_complexity': msg_complexity
            }
            
        except Exception as e:
            print(f"Error en análisis de dinámica conversacional: {e}")
            return {
                'depth': min(context.get('depth', 0) + 1, 10),
                'engagement': 5.0,
                'initiative_ratio': 0.5,
                'topic_continuity': 0,
                'topic_shift': True,
                'current_topic': 'general',
                'last_topics': ['general'],
                'topic_retention': 0
            }

   
    # 6. Selección avanzada de recursos relevantes
    def get_relevant_resources(user_message, context, limit_reflexiones=3, limit_books=2):
        """
        Sistema avanzado de recuperación de recursos relevantes con vectorización semántica
        """
        try:
            # Si no hay mensaje de usuario, seleccionar al azar
            if not user_message:
                return {
                    'reflexiones': Reflexion.query.order_by(db.func.random()).limit(limit_reflexiones).all(),
                    'books': Book.query.order_by(db.func.random()).limit(limit_books).all()
                }
            
            # Extracción de temas para mejorar la búsqueda
            topics = extract_topics(user_message)
            intent = detect_intent(user_message)
            
            # Vectorización simple de palabras clave (en producción, usar embeddings)
            palabras_clave = []
            
            # Añadir palabras clave de temas primarios
            for topic in topics['primary']:
                if topic in topic_taxonomy:
                    palabras_clave.extend(topic_taxonomy[topic]['keywords'])
            
            # Añadir palabras clave de temas secundarios
            for topic in topics['secondary']:
                if '.' in topic:  # Formato "principal.secundario"
                    main, sub = topic.split('.')
                    if main in topic_taxonomy and sub in topic_taxonomy[main]['subtopics']:
                        palabras_clave.extend(topic_taxonomy[main]['subtopics'][sub])
            
            # Añadir palabras del mensaje original (eliminando stopwords)
            stopwords = {'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'a', 'de', 'en', 'que', 'por', 'con'}
            palabras_originales = [palabra.lower() for palabra in user_message.split() 
                                 if len(palabra) > 3 and palabra.lower() not in stopwords]
            
            palabras_clave.extend(palabras_originales)
            
            # Eliminar duplicados y normalizar
            palabras_clave = list(set(palabras_clave))
            
            # Función de puntuación basada en relevancia
            def score_reflexion(reflexion):
                titulo_lower = reflexion.titulo.lower()
                contenido_lower = reflexion.contenido.lower()
                
                # Puntuación por presencia de palabras clave
                keyword_score = sum(2 if palabra in titulo_lower else 1 
                                   for palabra in palabras_clave 
                                   if palabra in titulo_lower or palabra in contenido_lower)
                
                # Bonus por relevancia temática
                topic_score = 0
                for topic in topics['primary']:
                    if topic in reflexion.tags:
                        topic_score += 3
                        
                # Bonus por intención del usuario
                intent_score = 0
                if intent['primary'] in ['pregunta_abierta', 'solicitud_info'] and 'informativo' in reflexion.tags:
                    intent_score += 2
                elif intent['primary'] in ['solicitud_consejo'] and 'consejos' in reflexion.tags:
                    intent_score += 3
                elif intent['primary'] in ['expresión_emocional'] and 'emocional' in reflexion.tags:
                    intent_score += 3
                    
                # Bonus por contexto conversacional
                context_score = 0
                if context.get('depth', 0) > 6 and 'profundo' in reflexion.tags:
                    context_score += 2
                elif context.get('depth', 0) < 3 and 'básico' in reflexion.tags:
                    context_score += 2
                
                # Freshness - preferir contenido no mostrado recientemente
                recency_score = 0
                if hasattr(reflexion, 'last_shown'):
                    days_since_shown = (datetime.datetime.now() - reflexion.last_shown).days
                    recency_score = min(5, days_since_shown)
                else:
                    recency_score = 5  # Máximo para contenido nunca mostrado
                
                return keyword_score + topic_score + intent_score + context_score + recency_score
            
            # Función similar para libros
            def score_book(book):
                title_lower = book.title.lower()
                content_lower = book.content.lower()
                
                keyword_score = sum(2 if palabra in title_lower else 1 
                                   for palabra in palabras_clave 
                                   if palabra in title_lower or palabra in content_lower)
                
                # Puntuación adicional similar a reflexiones
                topic_score = sum(2 for topic in topics['primary'] if topic in book.tags)
                
                return keyword_score + topic_score
            
            # Buscar y puntuar todas las reflexiones y libros
            reflexiones_scores = [(reflexion, score_reflexion(reflexion)) 
                                 for reflexion in Reflexion.query.all()]
            
            books_scores = [(book, score_book(book)) 
                           for book in Book.query.all()]
            
            # Ordenar por puntuación y seleccionar los mejores
            reflexiones_relevantes = [item[0] for item in sorted(reflexiones_scores, 
                                                               key=lambda x: x[1], 
                                                               reverse=True)[:limit_reflexiones]]
            
            books_relevantes = [item[0] for item in sorted(books_scores, 
                                                         key=lambda x: x[1], 
                                                         reverse=True)[:limit_books]]
            
            # Actualizar timestamp de última visualización
            for reflexion in reflexiones_relevantes:
                reflexion.last_shown = datetime.datetime.now()
                db.session.add(reflexion)
            
            db.session.commit()
            
            return {
                'reflexiones': reflexiones_relevantes,
                'books': books_relevantes
            }
            
        except Exception as e:
            print(f"Error al buscar recursos relevantes: {e}")
            return {
                'reflexiones': Reflexion.query.order_by(db.func.random()).limit(limit_reflexiones).all(),
                'books': Book.query.order_by(db.func.random()).limit(limit_books).all()
            }

    # 7. Gestión de preferencias de usuario adaptativa
    def update_user_preferences(user_id, message, context):
        """
        Actualiza las preferencias del usuario basadas en el mensaje y comportamiento
        """
        try:
            preferences = session.get('user_preferences', {})
            
            # Detector de preferencias de estilo de comunicación
            style_keywords = {
                'formal': ['formal', 'profesional', 'académico', 'técnico', 'riguroso'],
                'casual': ['casual', 'relajado', 'coloquial', 'informal', 'cercano'],
                'empático': ['empático', 'comprensivo', 'sensible', 'emocional', 'cálido'],
                'analítico': ['analítico', 'lógico', 'racional', 'estructurado', 'sistemático']
            }
            
            msg_lower = message.lower()
            
            # Detectar preferencias explícitas
            for style, keywords in style_keywords.items():
                for keyword in keywords:
                    if f"más {keyword}" in msg_lower or f"habla {keyword}" in msg_lower:
                        preferences['communication_style'] = style
                        break
            
            # Detectar preferencias de profundidad
            depth_keywords = {
                'conciso': ['breve', 'corto', 'conciso', 'resumido', 'directo'],
                'moderado': ['moderado', 'balanceado', 'equilibrado'],
                'profundo': ['profundo', 'detallado', 'extenso', 'elaborado', 'completo']
            }
            
            for depth, keywords in depth_keywords.items():
                for keyword in keywords:
                    if f"sé {keyword}" in msg_lower or f"responde {keyword}" in msg_lower:
                        preferences['response_depth'] = depth
                        break
            
            # Actualizar intereses detectados
            topics = extract_topics(message)
            current_interests = preferences.get('interests', [])
            
            # Añadir nuevos temas de interés
            for topic in topics['primary']:
                if topic not in current_interests and topic != 'general':
                    current_interests.append(topic)
            
            # Limitar a los 5 intereses más recientes
            preferences['interests'] = current_interests[-5:]
            
            # Actualizar preferencias cognitivas
            cognitive_style = analyze_cognitive_style(message)
            current_cognitive = preferences.get('cognitive_preferences', [])
            
            if cognitive_style['primary_style'] not in current_cognitive:
                current_cognitive.append(cognitive_style['primary_style'])
                
            # Limitar a los 3 estilos cognitivos más recientes
            preferences['cognitive_preferences'] = current_cognitive[-3:]
            
            # Actualizar en la sesión
            session['user_preferences'] = preferences
            
            # Si el usuario está autenticado, también actualizar en base de datos
            if user_id:
                user = User.query.get(user_id)
                if user:
                    user.preferences = json.dumps(preferences)
                    db.session.add(user)
                    db.session.commit()
            
            return preferences
            
        except Exception as e:
            print(f"Error al actualizar preferencias: {e}")
            return session.get('user_preferences', {})

    # Procesamiento de solicitudes POST (interacción con la IA)
    if request.method == 'POST':
        # Verificar límite de interacciones con advertencia proactiva
        if not (is_premium or is_admin) and remaining_interactions <= 2 and remaining_interactions > 0:
            warning = (
                f"¡Cuidado! Te quedan {remaining_interactions} interacciones hoy. "
                "Considera nuestra versión Premium para conversaciones ilimitadas."
            )
            return jsonify({
                'response': warning,
                'remaining_interactions': remaining_interactions,
                'limit_reached': False,
                'subscribe_url': url_for('subscribe_info', _external=True)
            })

        if not (is_premium or is_admin) and interaction_count >= effective_limit:
            return jsonify({
                'response': 'Has alcanzado tu límite diario de conversación.\n\n'
                            'Te invito a explorar nuestra versión Premium que ofrece:\n'
                            '- Conversaciones ilimitadas\n'
                            '- Acceso a recursos exclusivos\n'
                            '- Análisis de personalidad avanzado\n'
                            '- Retroalimentación personalizada\n\n'
                            '¡Tu crecimiento personal no tiene por qué detenerse!',
                'remaining_interactions': remaining_interactions,
                'limit_reached': True,
                'subscribe_url': url_for('subscribe_info', _external=True)
            })

        # Obtener el mensaje del usuario
        message = request.json.get('message', '').strip()
        
        # Análisis completo del mensaje y contexto
        start_time = time.time()
        
        # 1. Actualizar contexto temporal
        temporal_context = session.get('temporal_context', {})
        temporal_context['last_interaction_time'] = time.time()
        
        # Detectar hora especificada por el usuario
        try:
            hour_match = re.search(r'son las (\d{1,2}):(\d{2})(?:\s*hs)?', message.lower())
            if hour_match:
                hour = int(hour_match.group(1))
                minute = int(hour_match.group(2))
                temporal_context['user_specified_hour'] = hour + minute / 60
        except Exception as e:
            print(f"Error al detectar hora: {e}")
        
        session['temporal_context'] = temporal_context
        
        # 2. Análisis emocional
        emotional_context = session.get('emotional_context', {})
        current_emotion = analyze_sentiment(message, emotional_context.get('history', []))
        
        # Actualizar historial de emociones
        emotion_history = emotional_context.get('history', [])
        emotion_history.append(current_emotion)
        emotional_context['history'] = emotion_history[-5:]  # Mantener solo las 5 más recientes
        
        # Actualizar estado emocional actual
        emotional_context.update({
            'current_state': current_emotion['state'],
            'valence': current_emotion['valence'],
            'intensity': current_emotion['intensity'],
            'arousal': current_emotion['arousal'],
            'trend': current_emotion['trend']
        })
        
        session['emotional_context'] = emotional_context
        
        # 3. Actualizar métricas de conversación
        conversation_dynamics = session.get('conversation_dynamics', {})
        updated_dynamics = analyze_conversation_dynamics(
            message, 
            session.get('chat_history', []), 
            conversation_dynamics
        )
        conversation_dynamics.update(updated_dynamics)
        session['conversation_dynamics'] = conversation_dynamics
        
        # 4. Actualizar preferencias de usuario
        user_preferences = update_user_preferences(
            identifier, 
            message, 
            session.get('conversation_context', {})
        )
        
        # 5. Actualizar contexto de conversación
        conversation_context = session.get('conversation_context', {})
        conversation_context.update({
            'current_topic': updated_dynamics['current_topic'],
            'topic_retention': updated_dynamics['topic_retention'],
            'last_topics': updated_dynamics['last_topics'],
            'last_intent': updated_dynamics['last_intent'],
            'cognitive_load': min(10, conversation_context.get('cognitive_load', 0) + 
                                (1 if updated_dynamics['message_complexity'] > 2 else 0))
        })
        session['conversation_context'] = conversation_context
        
        # 6. Actualizar historial y contadores
        chat_history = session.get('chat_history', [])
        chat_history.append(f"Usuario: {message}")
        session['chat_history'] = chat_history[-20:]  # Mantener solo los 20 mensajes más recientes
        
        session_metrics = session.get('session_metrics', {})
        session_metrics['message_count'] = session_metrics.get('message_count', 0) + 1
        # Calcular calidad de interacción
        quality_score = min(10, (
            conversation_dynamics['engagement'] * 0.4 +
            conversation_dynamics['depth'] * 0.3 +
            conversation_dynamics['topic_continuity'] * 0.3
        ))
        session_metrics['interaction_quality'] = round(quality_score, 1)
        session['session_metrics'] = session_metrics
        
        # 7. Buscar recursos relevantes
        relevant_resources = get_relevant_resources(
            message,
            conversation_context,
            limit_reflexiones=3,
            limit_books=2
        )
        
        # Preparar resúmenes de recursos
        reflexiones_summary = "\n\n".join([
            f"Título: {reflexion.titulo}\n"
            f"Contenido: {reflexion.contenido[:300]}...\n"
            f"Relevancia: Alta\n"
            f"Tags: {', '.join(reflexion.tags)}"
            for reflexion in relevant_resources['reflexiones']
        ]) if relevant_resources['reflexiones'] else "No hay reflexiones relevantes para este tema."
        
        books_summary = "\n\n".join([
            f"Título: {book.title}\n"
            f"Contenido: {book.content[:300]}...\n"
            f"Relevancia: Alta\n"
            f"Tags: {', '.join(book.tags)}"
            for book in relevant_resources['books']
        ]) if relevant_resources['books'] else "No hay libros relevantes para este tema."

        # Calcular contexto temporal enriquecido
        try:
            current_time = datetime.datetime.now(tz=datetime.timezone(timedelta(hours=-3)))  # ART
            server_time_str = current_time.strftime("%H:%M")
            
            time_context = f"Hora del servidor: {server_time_str}"
            if temporal_context.get('user_specified_hour') is not None:
                time_context += f"\nHora especificada por el usuario: {temporal_context['user_specified_hour']:.2f}"
        except Exception as e:
            print(f"Error calculando contexto temporal: {e}")
            time_context = "Contexto temporal no disponible"
        
        # Componer prompt para la IA
        prompt = f"""
        [Contexto Conversacional]
        Tema actual: {conversation_context['current_topic']}
        Profundidad: {conversation_dynamics['depth']}/10
        Compromiso: {conversation_dynamics['engagement']}/10
        Intención del usuario: {conversation_context['last_intent']}
        Contexto temporal: {time_context}
        Sesión: {session_metrics['message_count']} mensajes en esta sesión

        [Contexto Emocional]
        Estado: {emotional_context['current_state']}
        Valencia: {emotional_context['valence']} (-5 a +5)
        Intensidad: {emotional_context['intensity']}/10
        Tendencia: {emotional_context['trend']}

        [Preferencias del Usuario]
        Estilo de comunicación: {user_preferences.get('communication_style', 'auto')}
        Profundidad de respuesta: {user_preferences.get('response_depth', 'adaptive')}
        Intereses: {', '.join(user_preferences.get('interests', ['no detectados']))}
        Preferencias cognitivas: {', '.join(user_preferences.get('cognitive_preferences', ['no detectadas']))}

        [Recursos Relevantes]
        {reflexiones_summary}

        [Mensaje del Usuario]
        {message}

        Responde de manera natural, adaptando tu estilo y profundidad según las preferencias del usuario y el contexto detectado. Toma en cuenta la profundidad de la conversación y el estado emocional detectado. Si el usuario está en un estado emocional negativo, muestra empatía. Si está en un estado positivo, refuerza esa energía. No menciones los detalles técnicos del análisis, solo úsalos para adaptar tu respuesta.
        """

        # Enviar solicitud a la API de IA
        try:
            # En producción, usar API real de modelos LLM
            # response = llm_api.generate(prompt)
            
            # Versión de demostración (simular respuesta)
            # Esto se reemplazaría por una llamada real a un LLM
            time.sleep(0.5)  # Simular tiempo de procesamiento
            
            if emotional_context['current_state'] in ['negative', 'distressed']:
                ia_response = f"Entiendo que esto puede ser un momento difícil. {message[:30]}... es algo que requiere atención. Las situaciones que nos generan emociones intensas suelen ser oportunidades de crecimiento. ¿Te gustaría explorar más sobre cómo manejar estos sentimientos?"
            elif emotional_context['current_state'] in ['positive', 'excited']:
                ia_response = f"¡Me alegra ver tu entusiasmo! {message[:30]}... refleja una energía positiva. Cuando nos conectamos con nuestras pasiones, abrimos puertas a nuevas posibilidades. ¿Qué aspectos de esto te generan más emoción?"
            else:
                topics = list(extract_topics(message)['primary'].keys())
                if topics and topics[0] != 'general':
                    ia_response = f"Interesante reflexión sobre {topics[0]}. {message[:30]}... plantea cuestiones que merecen ser exploradas. La perspectiva que compartes nos invita a considerar diferentes dimensiones de este tema. ¿Hay algún aspecto específico que te gustaría profundizar?"
                else:
                    ia_response = f"Gracias por compartir tus pensamientos. Lo que mencionas sobre {message[:20]}... ofrece una perspectiva valiosa. En momentos de reflexión como este, podemos descubrir nuevas capas de comprensión. ¿Hay algo más específico que te gustaría explorar?"
                    
            # En una implementación real, enviar el prompt completo a un modelo como GPT-4, Claude o similar
        except Exception as e:
            print(f"Error generando respuesta: {e}")
            ia_response = "Lo siento, estoy teniendo dificultades para procesar tu mensaje en este momento. ¿Podrías intentarlo de nuevo?"

        # Registrar la interacción
        try:
            interaction = Interaction(
                user_id=identifier,
                prompt=message,
                response=ia_response,
                sentiment=emotional_context['current_state'],
                topic=conversation_context['current_topic'],
                depth=conversation_dynamics['depth'],
                timestamp=datetime.datetime.now()
            )
            db.session.add(interaction)
            db.session.commit()
        except Exception as e:
            print(f"Error registrando interacción: {e}")
            db.session.rollback()

        # Actualizar historial
        chat_history.append(f"IA: {ia_response}")
        session['chat_history'] = chat_history[-20:]

        # Devolver respuesta
        return jsonify({
            'response': ia_response,
            'remaining_interactions': remaining_interactions - 1,
            'limit_reached': False,
            'emotion': emotional_context['current_state'],
            'topic': conversation_context['current_topic'],
            'depth': conversation_dynamics['depth'],
            'feedback_request': "Si te gustó esta respuesta, ¿podrías decirme qué te pareció útil?"
        })

    # Renderizar la página en solicitudes GET
    return render_template(
        'consciencia.html',
        user=current_user,
        remaining=remaining_interactions,
        total_limit=effective_limit,
        is_premium=is_premium
    )

@app.route('/subscribe_info')
def subscribe_info():
    return render_template('subscribe_info.html')

@app.route('/subscribe')
@login_required
def subscribe():
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': 'price_1R0K8OHB5KYU2SNZGSLPXsuE',
                'quantity': 1,
            }],
            mode='subscription',
            success_url=url_for('subscription_success', _external=True),
            cancel_url=url_for('cancel', _external=True),
        )
        return redirect(session.url)
    except stripe.error.StripeError as e:
        print(f"Error de Stripe: {e}")
        flash(f"Error al procesar la suscripción: {str(e)}", "error")
        return redirect(url_for('mostrar_consciencia'))

@app.route('/subscription_success')
@login_required
def subscription_success():
    user = current_user
    user.is_premium = True
    db.session.commit()
    flash("¡Suscripción exitosa! Ahora eres usuario premium.", "success")
    return render_template('subscription_success.html')

@app.route('/cancel')
def cancel():
    flash("La suscripción fue cancelada.", "warning")
    return redirect(url_for('mostrar_consciencia'))

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
            users = User.query.filter_by(frequency=frequency).all()
            if users:
                if frequency == 'daily':
                    scheduler.add_job(func=send_weekly_reflection, trigger='interval', days=1, id=f'reflection_{frequency}')
                elif frequency == 'weekly':
                    scheduler.add_job(func=send_weekly_reflection, trigger='interval', weeks=1, id=f'reflection_{frequency}')
                elif frequency == 'monthly':
                    scheduler.add_job(func=send_weekly_reflection, trigger='interval', weeks=4, id=f'reflection_{frequency}')

@app.route('/admin/users', methods=['GET', 'POST'])
@login_required
def admin_users():
    if not current_user.is_admin:
        flash('No tienes permiso para acceder a esta página.', 'error')
        return redirect(url_for('home'))

    if request.method == 'POST':
        user_id = request.form.get('user_id')
        if not user_id:
            flash('No se proporcionó un ID de usuario para eliminar.', 'error')
            return redirect(url_for('admin_users'))

        user_to_delete = User.query.get(user_id)
        if not user_to_delete:
            flash('El usuario no existe.', 'error')
            return redirect(url_for('admin_users'))

        if user_to_delete.is_admin and user_to_delete.id == current_user.id:
            flash('No puedes eliminar tu propia cuenta de administrador.', 'error')
            return redirect(url_for('admin_users'))

        try:
            db.session.execute(text("DELETE FROM favorite WHERE user_id = :user_id"), {"user_id": user_id})
            db.session.delete(user_to_delete)
            db.session.commit()
            flash(f'Usuario {user_to_delete.email} eliminado exitosamente.', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error al eliminar usuario: {str(e)}', 'error')

        return redirect(url_for('admin_users'))

    users = User.query.all()
    return render_template('admin_users.html', users=users)

def migrate_subscriber_to_user():
    with app.app_context():
        if 'subscriber' in inspect(db.engine).get_table_names():
            old_subscribers = db.session.execute(
                text('SELECT email, subscription_date, preferred_categories, frequency, push_enabled, onesignal_player_id FROM subscriber')
            ).fetchall()
            for sub in old_subscribers:
                user = User.query.filter_by(email=sub[0]).first()
                if user:
                    user.subscription_date = sub[1]
                    user.preferred_categories = sub[2] if sub[2] else 'all'
                    user.frequency = sub[3] if sub[3] else 'weekly'
                    user.push_enabled = sub[4] if sub[4] is not None else False
                    user.onesignal_player_id = sub[5]
                else:
                    user = User(
                        email=sub[0],
                        subscription_date=sub[1],
                        preferred_categories=sub[2] if sub[2] else 'all',
                        frequency=sub[3] if sub[3] else 'weekly',
                        push_enabled=sub[4] if sub[4] is not None else False,
                        onesignal_player_id=sub[5],
                        password_hash=generate_password_hash('default_password'),
                        is_active=True
                    )
                    db.session.add(user)
            db.session.commit()
            print("Datos migrados de Subscriber a User")
            db.session.execute(text('DROP TABLE subscriber'))
            db.session.commit()
            print("Tabla Subscriber eliminada")
        else:
            print("No se encontró la tabla Subscriber para migrar")

if __name__ == '__main__':
    with app.app_context():
        print("Tablas gestionadas por Alembic")
        db.create_all()
        load_pdfs_to_db()

    scheduler = BackgroundScheduler()
    schedule_reflections()
    scheduler.start()
    app.run(
        host='0.0.0.0',
        port=5001,
        ssl_context=('cert.pem', 'key.pem'),
        debug=True  # Cambiado a False para pruebas
    )