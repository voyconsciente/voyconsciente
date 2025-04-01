import json
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, make_response, session
from flask_mail import Mail, Message
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from flask_assets import Environment, Bundle
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from weasyprint import HTML
from apscheduler.schedulers.background import BackgroundScheduler
import io
import datetime
from datetime import datetime, timedelta, timezone
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
import re

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar variables de entorno desde .env si existe (local), pero no fallar si no está
if os.path.exists('.env'):
    load_dotenv()
else:
    print("Archivo .env no encontrado. Usando variables de entorno del sistema (esperado en producción).")

# Configuración de la sesión permanente
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_NAME'] = 'voy_session'

# Configuración de la clave secreta
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    raise ValueError("Error: No se encontró SECRET_KEY en las variables de entorno. Configúrala en .env o en el entorno de producción.")

# Configuración de Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Definir la función is_safe_url globalmente
def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    dest_url = urlparse(urljoin(request.host_url, target))
    return dest_url.scheme in ('http', 'https') and ref_url.netloc == dest_url.netloc

# Inicializar el serializador para generar tokens seguros
serializer = URLSafeTimedSerializer(app.secret_key)

# Configuración de Google Analytics
GA_CREDENTIALS_JSON = os.getenv('GA_CREDENTIALS_JSON')
if not GA_CREDENTIALS_JSON:
    raise ValueError("Error: No se encontró GA_CREDENTIALS_JSON en las variables de entorno.")
credentials_dict = json.loads(GA_CREDENTIALS_JSON)
credentials = service_account.Credentials.from_service_account_info(credentials_dict)
analytics_client = BetaAnalyticsDataClient(credentials=credentials)
GA_PROPERTY_ID = os.getenv('GA_PROPERTY_ID', '480922494')
GA_FLOW_ID = os.getenv('GA_FLOW_ID', '10343079148')
GA_FLOW_NAME = os.getenv('GA_FLOW_NAME', 'Voy Consciente')
GA_FLOW_URL = os.getenv('GA_FLOW_URL', 'https://192.168.0.213:5001')

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
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Asegúrate de que el resto del código permanezca igual
with app.app_context():
    db.create_all()

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
    memory_context = db.Column(db.Text, nullable=True)

    @staticmethod
    def get_interaction_count(identifier, is_authenticated):
        today = datetime.now(tz=timezone(timedelta(hours=-3))).date()
        if is_authenticated:
            interaction = Interaction.query.filter_by(user_id=identifier, interaction_date=today).first()
        else:
            interaction = Interaction.query.filter_by(session_id=identifier, interaction_date=today).first()

        if not interaction:
            interaction = Interaction(
                user_id=identifier if is_authenticated else None,
                session_id=identifier if not is_authenticated else None,
                interaction_date=today,
                interaction_count=0,
                memory_context=json.dumps([])
            )
            db.session.add(interaction)
            db.session.commit()

        return interaction.interaction_count

    @staticmethod
    def increment_interaction(identifier, is_authenticated, user_message=None, ai_response=None):
        today = datetime.now(tz=timezone(timedelta(hours=-3))).date()
        if is_authenticated:
            interaction = Interaction.query.filter_by(user_id=identifier, interaction_date=today).first()
        else:
            interaction = Interaction.query.filter_by(session_id=identifier, interaction_date=today).first()

        if not interaction:
            interaction = Interaction(
                user_id=identifier if is_authenticated else None,
                session_id=identifier if not is_authenticated else None,
                interaction_date=today,
                interaction_count=0,
                memory_context=json.dumps([])
            )
        interaction.interaction_count += 1

        if user_message and ai_response:
            memory = json.loads(interaction.memory_context or '[]')
            memory.append({"user": user_message, "ai": ai_response, "timestamp": datetime.now().isoformat()})
            interaction.memory_context = json.dumps(memory[-10:])
        
        db.session.add(interaction)
        db.session.commit()
        return interaction.interaction_count

    @staticmethod
    def get_memory_context(identifier, is_authenticated):
        today = datetime.now(tz=timezone(timedelta(hours=-3))).date()
        if is_authenticated:
            interaction = Interaction.query.filter_by(user_id=identifier, interaction_date=today).first()
        else:
            interaction = Interaction.query.filter_by(session_id=identifier, interaction_date=today).first()
        if interaction and interaction.memory_context:
            return json.loads(interaction.memory_context)
        return []

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
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(tz=timezone(timedelta(hours=-3))))

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

# Función para interactuar con la API de Grok
def call_grok_api(messages, max_tokens=80, temperature=0.8):
    try:
        payload = {
            "messages": messages,
            "model": "grok-2-latest",
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9
        }
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {grok_api_key}", "Content-Type": "application/json"},
            json=payload
        )
        response.raise_for_status()
        response_json = response.json()
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "Ups, algo salió mal.")
    except requests.RequestException as e:
        print(f"Error en la solicitud a xAI: {str(e)}")
        return "Ups, algo salió mal. ¡Intentemos de nuevo!"

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

@app.route('/terminos-condiciones')
def terminos_condiciones():
    return render_template('terminos_condiciones.html', año_actual=datetime.now().year)

@app.route('/politica-privacidad')
def politica_privacidad():
    return render_template('politica_privacidad.html', año_actual=datetime.now().year)

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
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, 'js/main.js')
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
    return render_template('index.html', año_actual=datetime.now().year)

@app.route('/sobre-nosotros')
def sobre_nosotros():
    return render_template('sobre_nosotros.html')

@app.route('/reflexiones', defaults={'page': 1})
@app.route('/reflexiones/page/<int:page>')
@cache.cached(timeout=300, key_prefix=lambda: f"reflexiones_page_{request.args.get('page', 1)}")
def mostrar_reflexiones(page):
    per_page = 20
    reflexiones = Reflexion.query.paginate(page=page, per_page=per_page, error_out=False)
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
            session.permanent = True
            login_user(user, remember=True)
            if next_page and is_safe_url(next_page):
                return redirect(next_page)
            return redirect(url_for('home'))
        else:
            flash('Correo o contraseña incorrectos.', 'error')
            return render_template('login.html', next=next_page, email=email)

    next_page = request.args.get('next', '')
    email = request.args.get('email', '')
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

        try:
            birth_date = datetime.strptime(birth_date_str, '%d/%m/%Y')
            birth_date_iso = birth_date.isoformat()
        except ValueError:
            flash('El formato de la fecha de nacimiento debe ser DD/MM/AAAA.', 'error')
            return render_template('register.html', next=next_page, name=name, email=email, birth_date=birth_date_str, phone=phone, password=password, confirm_password=confirm_password)

        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'
        if password != confirm_password:
            flash('Las contraseñas no coinciden.', 'error')
            return render_template('register.html', next=next_page, name=name, email=email, birth_date=birth_date_str, phone=phone, password=password, confirm_password=confirm_password)
        if not re.match(password_pattern, password):
            flash('La contraseña debe tener al menos 8 caracteres, una mayúscula, una minúscula y un número.', 'error')
            return render_template('register.html', next=next_page, name=name, email=email, birth_date=birth_date_str, phone=phone, password=password, confirm_password=confirm_password)

        if User.query.filter_by(email=email).first():
            flash('El correo ya está registrado.', 'error')
            return render_template('register.html', next=next_page, name=name, email=email, birth_date=birth_date_str, phone=phone, password=password, confirm_password=confirm_password)

        user = User(
            email=email,
            subscription_date=datetime.now().isoformat(),
            preferred_categories='all',
            frequency='weekly',
            birth_date=birth_date_iso
        )
        user.set_password(password)
        user.is_active = False
        token = user.generate_activation_token()
        db.session.add(user)
        db.session.commit()

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
                          año_actual=datetime.now(tz=timezone(timedelta(hours=-3))).year)
    pdf_file = io.BytesIO()
    HTML(string=html).write_pdf(pdf_file)
    pdf_file.seek(0)
    response = make_response(pdf_file.read())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename="{reflexion.titulo}.pdf"'
    return response

@app.route('/consciencia', methods=['GET', 'POST'])
@login_required
def mostrar_consciencia():
    identifier = current_user.id
    is_premium = current_user.is_premium
    is_admin = current_user.is_admin
    FREE_INTERACTION_LIMIT = 5
    interaction_count = Interaction.get_interaction_count(identifier, True)
    remaining_interactions = 999999 if (is_premium or is_admin) else max(0, FREE_INTERACTION_LIMIT - interaction_count)

    # Obtener memoria contextual
    memory_context = Interaction.get_memory_context(identifier, True)
    user_name = None
    for entry in memory_context:
        name_match = re.search(r"(me llamo|mi nombre es)\s+([a-zA-Z]+)", entry.get("user", "").lower())
        if name_match:
            user_name = name_match.group(2).capitalize()
            break

    if request.method == 'GET':
        greeting = f"¡Hola, {user_name}! Estoy aquí para conversar cuando gustes." if user_name else \
                   "¡Hola! Estoy listo para charlar contigo. ¿Cómo te gustaría empezar?"
        return render_template('consciencia.html', initial_response=greeting, remaining_interactions=remaining_interactions)

    if request.method == 'POST':
        try:
            message = request.json.get('message', '').strip()
            if not message or len(message) < 2:
                return jsonify({
                    'response': "Entiendo, cuando quieras decir algo más, aquí estaré.",
                    'remaining_interactions': remaining_interactions - 1
                })

            # Detectar nombre si no se conoce
            if not user_name:
                name_match = re.search(r"(me llamo|mi nombre es)\s+([a-zA-Z]+)", message.lower())
                if name_match:
                    user_name = name_match.group(2).capitalize()
                    response_text = f"¡Encantado de conocerte, {user_name}! Me alegra charlar contigo."
                    Interaction.increment_interaction(identifier, True, message, response_text)
                    return jsonify({'response': response_text, 'remaining_interactions': remaining_interactions - 1})

            # Construir contexto del sistema con personalidad y memoria
            system_message = {
                "role": "system",
                "content": (
                    "Soy ConciencIA, creada por Voy Consciente. Soy empática, alegre y profesional. "
                    "Mi propósito es escuchar y charlar contigo con calidez y respeto, siguiendo tu ritmo. "
                    f"Usuario: {user_name or 'estimado usuario'}. "
                    "Tono: cortés, optimista y claro. Responde en 2-3 oraciones máximo, evita preguntas a menos que sean esenciales."
                )
            }

            # Resumir memoria para eficiencia de tokens
            memory_summary = "Contexto reciente: " + " ".join(
                [f"{entry['user']} -> {entry['ai'][:30]}..." for entry in memory_context[-3:]]
            ) if memory_context else "Sin interacciones previas."

            # Construir mensajes para la API
            messages = [system_message]
            if memory_summary:
                messages.append({"role": "system", "content": memory_summary})
            messages.append({"role": "user", "content": message[:100]})

            # Llamar a la API de Grok
            response_text = call_grok_api(messages, max_tokens=80, temperature=0.8)

            # Guardar interacción en memoria
            Interaction.increment_interaction(identifier, True, message, response_text)

            return jsonify({
                'response': response_text.strip(),
                'remaining_interactions': remaining_interactions - 1,
                'limit_reached': remaining_interactions <= 1
            })

        except Exception as e:
            print(f"Error en /consciencia: {str(e)}")
            return jsonify({
                'response': "Lo siento, algo salió mal. Estoy aquí cuando quieras seguir.",
                'remaining_interactions': remaining_interactions - 1
            }), 500

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
                text('SELECT email, subscription_date, preferred_categories, frequency FROM subscriber')
            ).fetchall()
            for sub in old_subscribers:
                user = User.query.filter_by(email=sub[0]).first()
                if user:
                    user.subscription_date = sub[1]
                    user.preferred_categories = sub[2] if sub[2] else 'all'
                    user.frequency = sub[3] if sub[3] else 'weekly'
                else:
                    user = User(
                        email=sub[0],
                        subscription_date=sub[1],
                        preferred_categories=sub[2] if sub[2] else 'all',
                        frequency=sub[3] if sub[3] else 'weekly',
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
        debug=True
    )