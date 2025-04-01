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
from datetime import datetime, timedelta, timezone  # Reemplaza UTC por timezone
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

# Nuevas importaciones para la gestión de memoria vectorial
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os.path

# Nuevas importaciones para el motor de personalidad
from sklearn.cluster import KMeans
from scipy.stats import norm

# Nuevas importaciones para el sistema de meta-cognición
from collections import Counter

# Crear la aplicación Flask primero
app = Flask(__name__)

# Cargar variables de entorno desde .env
if not os.path.exists('.env'):
    raise FileNotFoundError("Archivo .env no encontrado. Asegúrate de crearlo y configurarlo correctamente.")
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuración de la sesión permanente
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_NAME'] = 'voy_session'

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
    return db.session.get(User, int(user_id))

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
    name = db.Column(db.String(50), nullable=True)    
    timezone_offset = db.Column(db.Integer, default=-3)
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
        today = datetime.now(tz=timezone(timedelta(hours=-3))).date()  # UTC-3
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
        today = datetime.now(tz=timezone(timedelta(hours=-3))).date()  # UTC-3
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

# Configuración de OneSignal
onesignal_client = OneSignalClient(
    app_id=os.getenv("ONESIGNAL_APP_ID"),
    rest_api_key=os.getenv("ONESIGNAL_REST_API_KEY")
)
if not onesignal_client.app_id or not onesignal_client.rest_api_key:
    raise ValueError("Error: Faltan ONESIGNAL_APP_ID o ONESIGNAL_REST_API_KEY en las variables de entorno.")

# Clase para gestión de memoria vectorial
from datetime import datetime, timedelta, timezone
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re

class UserMemoryManager:
    def __init__(self, user_id, model_name='all-MiniLM-L6-v2'):
        self.user_id = user_id
        self.memory_file = f"user_memories/{user_id}_memory.pkl"
        self.index_file = f"user_memories/{user_id}_index.faiss"
        self.cluster_file = f"user_memories/{user_id}_clusters.pkl"
        self.model = SentenceTransformer(model_name)
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        
        self.memories = {
            'interactions': [],
            'topics': {},
            'preferences': {},
            'episodic_memories': [],
            'last_session': None,
            'clusters': {}
        }
        
        self.short_term_buffer = []
        self.attention_weights = {}
        
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.cluster_model = None
        self.load_memory()

     
    def load_memory(self):
        """Carga memoria y verifica el nombre desde la DB si no está en preferencias"""
        default_memories = {
            'interactions': [],
            'topics': {},
            'preferences': {},
            'episodic_memories': [],
            'last_session': None,
            'clusters': {}
        }
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    loaded_memories = pickle.load(f)
                    self.memories = default_memories.copy()
                    self.memories.update(loaded_memories)
                    # Asegurar timestamps aware
                    for interaction in self.memories['interactions']:
                        if 'timestamp' in interaction and isinstance(interaction['timestamp'], datetime) and interaction['timestamp'].tzinfo is None:
                            interaction['timestamp'] = interaction['timestamp'].replace(tzinfo=timezone(timedelta(hours=-3)))
                    if self.memories['last_session'] and isinstance(self.memories['last_session'], datetime) and self.memories['last_session'].tzinfo is None:
                        self.memories['last_session'] = self.memories['last_session'].replace(tzinfo=timezone(timedelta(hours=-3)))
            else:
                self.memories = default_memories.copy()
            # Verificar nombre desde User si no está en preferencias
            if 'name' not in self.memories['preferences']:
                user = User.query.get(self.user_id)
                if user and hasattr(user, 'name') and user.name:
                    self.update_preference('name', user.name)
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
            else:
                self.index = faiss.IndexFlatL2(self.vector_dim)
            if os.path.exists(self.cluster_file):
                with open(self.cluster_file, 'rb') as f:
                    self.cluster_model = pickle.load(f)
        except Exception as e:
            print(f"Error al cargar memoria: {e}")
            self.memories = default_memories.copy()

    def get_user_name(self):
        """Obtiene el nombre del usuario desde preferencias"""
        return self.memories['preferences'].get('name', {}).get('value', None)

    def set_user_name(self, name):
        """Guarda el nombre en preferencias y en la base de datos"""
        self.update_preference('name', name)
        user = User.query.get(self.user_id)
        if user:
            user.name = name
            db.session.commit()

    
# Nueva clase PersonalityEngine
class PersonalityEngine:
    def __init__(self, user_id):
        self.user_id = user_id
    def generate_instruction_set(self, conversation_depth=1, context=None):
        context = context or {}
        emotion = context.get('emoción', 'neutral')
        tone = {
            'happy': "alegre y con chispa",
            'sad': "cálido y reconfortante",
            'angry': "suave y comprensivo",
            'anxious': "tranquilo y apoyo",
            'neutral': "relajado y simpático"
        }.get(emotion, "relajado y simpático")
        return {
            'instruction_text': f"Charlá {tone}, con humor ligero o empatía según pinte.",
            'parameters': {'max_tokens': 100, 'temperature': 0.85}
        }
 

# Nueva clase MetaCognitionSystem
class MetaCognitionSystem:
    """Sistema de meta-cognición para permitir que ConciencIA evalúe y mejore sus respuestas"""
    
    def __init__(self, user_id, custom_thresholds=None, custom_weights=None):
        self.user_id = user_id
        self.meta_file = f"user_metacognition/{user_id}_metacog.json"
        self._changes_since_save = 0
        
        # Métricas de auto-evaluación
        self.evaluation_metrics = {
            'relevance': [],          # Relevancia de respuestas anteriores
            'coherence': [],          # Coherencia lógica interna
            'helpfulness': [],        # Utilidad percibida
            'depth': [],              # Profundidad conceptual
            'engagement': [],         # Capacidad de mantener el interés
            'misunderstandings': [],  # Registro de malentendidos detectados
        }
        
        # Umbrales adaptativos para la auto-mejora (configurables)
        self.improvement_thresholds = custom_thresholds or {
            'relevance_threshold': 0.7,
            'coherence_threshold': 0.8,
            'elaboration_needed': 0.6,
            'correction_needed': 0.4,
        }
        
        # Patrones de diálogo problemáticos identificados
        self.dialog_patterns = {
            'repetitive_responses': 0,
            'topic_misalignments': 0,
            'question_avoidance': 0,
            'excessive_complexity': 0,
            'excessive_simplicity': 0,
        }
        
        # Cargar datos existentes
        self.load_metacognition()
    
    def load_metacognition(self):
        """Carga datos de meta-cognición si existen"""
        try:
            if os.path.exists(self.meta_file):
                with open(self.meta_file, 'r') as f:
                    data = json.load(f)
                    self.evaluation_metrics = data.get('evaluation_metrics', self.evaluation_metrics)
                    self.improvement_thresholds = data.get('improvement_thresholds', self.improvement_thresholds)
                    self.dialog_patterns = data.get('dialog_patterns', self.dialog_patterns)
        except Exception as e:
            print(f"Error al cargar meta-cognición: {e}")
    
    def save_metacognition(self):
        """Guarda datos de meta-cognición con debounce"""
        self._changes_since_save += 1
        if self._changes_since_save >= 5:  # Guardar cada 5 cambios
            try:
                os.makedirs("user_metacognition", exist_ok=True)
                with open(self.meta_file, 'w') as f:
                    json.dump({
                        'evaluation_metrics': self.evaluation_metrics,
                        'improvement_thresholds': self.improvement_thresholds,
                        'dialog_patterns': self.dialog_patterns
                    }, f)
                self._changes_since_save = 0
            except Exception as e:
                print(f"Error al guardar meta-cognición: {e}")
    
    def evaluate_response(self, user_message, ai_response, previous_messages=None, user_emotion='neutral', current_time=None):
        """Evalúa la calidad de la respuesta con predicción de necesidades"""
        if not previous_messages:
            previous_messages = []
        if not current_time:
            current_time = datetime.now(tz=timezone(timedelta(hours=-3)))
    
        scores = {
            'relevance': 0.0,
            'coherence': 0.0,
            'helpfulness': 0.0,
            'depth': 0.0,
            'engagement': 0.0,
            'need_satisfaction': 0.0
        }
    
        try:
            # Predicción de necesidades
            needs_pred = self.predict_user_needs(user_message, previous_messages, user_emotion, current_time)
            primary_need = needs_pred['primary_need']
        
            # 1. Relevancia
            user_keywords = self._extract_keywords(user_message)
            response_keywords = self._extract_keywords(ai_response)
            semantic_overlap = len(set(user_keywords) & set(response_keywords)) / max(1, len(set(user_keywords)))
            scores['relevance'] = min(1.0, semantic_overlap * 1.5)
        
            # 2. Coherencia
            contradiction_detected = self._detect_contradictions(ai_response)
            scores['coherence'] = 0.9 - (0.5 if contradiction_detected else 0.0)
        
            # 3. Profundidad
            word_count = len(ai_response.split())
            unique_words = len(set(ai_response.lower().split()))
            complexity_ratio = unique_words / max(1, word_count)
            long_words = len([w for w in ai_response.lower().split() if len(w) > 7])
            long_word_ratio = long_words / max(1, word_count)
            scores['depth'] = min(1.0, (complexity_ratio * 0.5) + (long_word_ratio * 2.0) + (word_count / 200 * 0.3))
        
            # 4. Engagement
            engagement_score = 0.5
            if '?' in ai_response:
                engagement_score += 0.2
            if any(marker in ai_response.lower() for marker in ['por ejemplo', 'ejemplo', 'como cuando', 'imagina']):
                engagement_score += 0.15
            if any(marker in ai_response.lower() for marker in ['es como', 'similar a', 'equivale a', 'se asemeja']):
             engagement_score += 0.15
            scores['engagement'] = min(1.0, engagement_score)
        
            # 5. Utilidad ajustada por necesidad
            if primary_need == 'information' and any(marker in user_message.lower() for marker in ['cómo', 'qué', 'dime', 'saber']):
                scores['helpfulness'] = 0.7 + (scores['relevance'] * 0.3) if scores['relevance'] > 0.6 else 0.4
            elif primary_need == 'emotional_support':
                scores['helpfulness'] = 0.6 + (scores['engagement'] * 0.4) if 'apoyo' in ai_response.lower() or 'entiendo' in ai_response.lower() else 0.3
            else:
                scores['helpfulness'] = 0.5 + (scores['relevance'] * 0.2) + (scores['engagement'] * 0.3)
        
            # 6. Satisfacción de necesidad
            need_markers = {
                'clarification': ['aclaro', 'explico', 'entiendo', 'claro'],
                'emotional_support': ['ánimo', 'entiendo', 'estoy aquí', 'tranquilo'],
                'information': ['dato', 'explicación', 'saber', 'aquí tienes'],
                'action_suggestion': ['sugiero', 'puedes', 'intenta', 'prueba']
            }
            response_lower = ai_response.lower()
            markers_found = sum(1 for marker in need_markers.get(primary_need, []) if marker in response_lower)
            scores['need_satisfaction'] = min(1.0, 0.5 + (markers_found * 0.2))  # Base 0.5, +0.2 por marcador
        
            # Actualizar patrones y métricas
            self._update_dialog_patterns(user_message, ai_response, previous_messages)
            for metric, value in scores.items():
                self.evaluation_metrics[metric].append(value)
                if len(self.evaluation_metrics[metric]) > 20:
                    self.evaluation_metrics[metric] = self.evaluation_metrics[metric][-20:]
            self.save_metacognition()
        
        except Exception as e:
            print(f"Error al evaluar respuesta: {e}")
            scores['need_satisfaction'] = 0.3  # Valor por defecto en caso de error
    
        return scores
    
    
    def generate_self_improvements(self, scores, user_message, ai_response, intent='unknown'):
        """Genera mejoras basadas en la evaluación de la respuesta"""
        improvements = {
            'adjusted_temperature': None,
            'suggested_prompt_additions': [],
            'focus_areas': []  # Agregamos esto para evitar el KeyError
        }
    
        # Ajustes por baja relevancia
        if scores.get('relevance', 0) < 0.5:
            improvements['suggested_prompt_additions'].append(
                "Asegúrate de responder directamente a lo que el usuario menciona."
            )
    
        # Ajustes por incoherencia
        if scores.get('coherence', 0) < 0.6:
            improvements['suggested_prompt_additions'].append(
                "Revisa que tus ideas sean consistentes y no te contradigas."
            )
    
        # Ajustes por baja profundidad o relevancia
        if scores.get('depth', 0) < 0.4:
            improvements['suggested_prompt_additions'].append(
                "Añade un poco más de detalle o una idea extra si encaja."
            )
            improvements['suggested_prompt_additions'].append(
                "Considerá buscar info externa (web o X) para sumar un dato fresco y enriquecer la respuesta."
            )
            improvements['focus_areas'].append('usar_búsqueda_externa')
                
        # Ajustes por bajo engagement
        if scores.get('engagement', 0) < 0.5:
            improvements['suggested_prompt_additions'].append(
                "Intenta enganchar más con una pregunta o un ejemplo sencillo."
            )
    
        # Ajustes por repeticiones más estrictos
        if self.dialog_patterns['repetitive_responses'] > 1:
            improvements['suggested_prompt_additions'].append(
                "Varía tu lenguaje y evita repetir frases o ideas de respuestas recientes."
            )
            improvements['adjusted_temperature'] = min(0.8, personality_instructions['parameters']['temperature'] + 0.1)
            self.dialog_patterns['repetitive_responses'] -= 1
    
        # Reforzar optimismo si la respuesta es útil
        if scores.get('helpfulness', 0) > 0.7 and scores.get('engagement', 0) > 0.6:
            improvements['suggested_prompt_additions'].append(
                "Sigue destacando un enfoque positivo y motivador; parece que está funcionando bien."
            )
        elif scores.get('helpfulness', 0) < 0.4:
            improvements['suggested_prompt_additions'].append(
                "Intenta añadir un toque más optimista o una idea práctica para ser más útil."
            )
    
        # Detectar exceso de confianza o falta de sensatez
        if 'todo estará bien' in ai_response.lower() or 'seguro que' in ai_response.lower():
            improvements['suggested_prompt_additions'].append(
                "Evita sonar demasiado seguro o prometedor; sé más humilde y práctico."
            )
        if scores.get('helpfulness', 0) < 0.5 and intent in ['information', 'action_suggestion']:
            improvements['suggested_prompt_additions'].append(
                "Da una idea más concreta y realista para que sea más útil."
            )
    
            self.save_metacognition()
            print(f"Mejoras sugeridas: {improvements}")  # Agrega esto antes del return
            return improvements

        avg_metrics = {k: sum(v[-5:]) / max(1, len(v[-5:])) for k, v in self.evaluation_metrics.items() if v}

        # Detectar instrucciones explícitas del usuario
        user_lower = user_message.lower()
        if "no me recuerdes" in user_lower or "no repitas" in user_lower:
            improvements['suggested_prompt_additions'].append(
                "No menciones recuerdos específicos del pasado a menos que el usuario lo pida explícitamente."
            )
            self.dialog_patterns['repetitive_responses'] += 1
        if "actúa más normal" in user_lower or "sé más natural" in user_lower:
            improvements['suggested_prompt_additions'].append(
                "Adopta un tono más casual y relajado, evitando exageraciones o referencias innecesarias."
            )
            improvements['adjusted_temperature'] = 0.6  # Más controlado
            self.dialog_patterns['excessive_complexity'] = max(0, self.dialog_patterns['excessive_complexity'] - 1)

        # Ajustes basados en métricas
        if avg_metrics.get('relevance', 0.7) < self.improvement_thresholds['relevance_threshold']:
            improvements['adjusted_temperature'] = 0.65
            improvements['focus_areas'].append('aumentar_relevancia')
        if avg_metrics.get('coherence', 0.8) < self.improvement_thresholds['coherence_threshold']:
            improvements['focus_areas'].append('mejorar_coherencia')

        if scores.get('depth', 0.5) < self.improvement_thresholds['elaboration_needed']:
            improvements['adjusted_max_tokens'] = 150  # Menos tokens para respuestas más cortas

        # Ajustes por repeticiones más estrictos
        if self.dialog_patterns['repetitive_responses'] > 1:
            improvements['suggested_prompt_additions'].append(
                "Varía tu lenguaje y evita repetir frases o ideas de respuestas recientes."
            )
            improvements['adjusted_temperature'] = min(0.8, personality_instructions['parameters']['temperature'] + 0.1)  # Más creatividad
            self.dialog_patterns['repetitive_responses'] -= 1
            
         # Reforzar optimismo si la respuesta es útil
        if scores.get('helpfulness', 0) > 0.7 and scores.get('engagement', 0) > 0.6:
            improvements['suggested_prompt_additions'].append(
                "Sigue destacando un enfoque positivo y motivador; parece que está funcionando bien."
            )
        elif scores.get('helpfulness', 0) < 0.4:
            improvements['suggested_prompt_additions'].append(
                "Intenta añadir un toque más optimista o una idea práctica para ser más útil."
            )   
            
        # Detectar exceso de confianza o falta de sensatez
        if 'todo estará bien' in ai_response.lower() or 'seguro que' in ai_response.lower():
            improvements['suggested_prompt_additions'].append(
                "Evita sonar demasiado seguro o prometedor; sé más humilde y práctico."
            )
        if scores.get('helpfulness', 0) < 0.5 and intent in ['information', 'action_suggestion']:
            improvements['suggested_prompt_additions'].append(
                "Da una idea más concreta y realista para que sea más útil."
            )

        self.save_metacognition()
        print(f"Mejoras sugeridas: {improvements}")  # Agrega esto
        return improvements
    
    def _extract_keywords(self, text):
        """Extrae palabras clave de un texto"""
        try:
            text = text.lower()
            words = text.split()
            stopwords = {'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'de', 'del', 'a', 'en', 'que', 'es', 'por', 'para', 'con', 'su', 'sus'}
            return [word for word in words if len(word) > 3 and word not in stopwords]
        except Exception as e:
            print(f"Error al extraer keywords: {e}")
            return []
    
    def _detect_contradictions(self, text):
        """Detecta posibles contradicciones internas en el texto"""
        try:
            contradiction_patterns = [
                (r'no .{1,20} pero .{1,20} sí', 'afirmación-negación'),
                (r'siempre .{1,30} nunca', 'absolutos contradictorios'),
                (r'es .{1,20} no es', 'afirmación-negación directa')
            ]
            for pattern, _ in contradiction_patterns:
                if re.search(pattern, text.lower()):
                    return True
            return False
        except Exception as e:
            print(f"Error al detectar contradicciones: {e}")
            return False
    
    def _update_dialog_patterns(self, user_message, ai_response, previous_messages):
        if len(previous_messages) >= 2:
            last_response = previous_messages[-1]
            repeated_phrases = self._find_repeated_phrases(last_response, ai_response)
            if repeated_phrases >= 1:  # Más estricto: 1 repetición ya es problema
                self.dialog_patterns['repetitive_responses'] += 1
                print(f"Repetición detectada: {repeated_phrases} frases repetidas")
        
        if '?' in user_message and '?' not in ai_response:
            question_words = ['cómo', 'qué', 'cuándo', 'dónde', 'por qué', 'cuál', 'quién']
            if any(word in user_message.lower() for word in question_words):
                self.dialog_patterns['question_avoidance'] += 1
        
        if previous_messages and len(previous_messages) >= 2:
            prev_user_message = previous_messages[-2]
            user_topics = set(self._extract_keywords(user_message))
            prev_topics = set(self._extract_keywords(prev_user_message))
            if len(user_topics & prev_topics) > 2:
                response_topics = set(self._extract_keywords(ai_response))
                if len(user_topics & response_topics) < 2:
                    self.dialog_patterns['topic_misalignments'] += 1
    
    def _find_repeated_phrases(self, text1, text2, min_phrase_length=5):
        """Encuentra frases repetidas entre dos textos"""
        words1 = text1.lower().split()
        words2 = set(text2.lower().split())
        return sum(1 for i in range(len(words1) - min_phrase_length + 1) if ' '.join(words1[i:i+min_phrase_length]) in ' '.join(words2))
    
    def _detect_misunderstanding(self, user_message, ai_response):
        """Detecta posibles malentendidos en la respuesta"""
        misunderstanding_patterns = [
            (r'\?.*\bcómo\b|\bqué\b|\bcuál\b|\bcuándo\b', r'', 'respuesta_general_a_pregunta_específica'),
            (r'\bno\b.*\b(quiero|me gusta|deseo)\b', r'\bperfecto\b|\bexcelente\b|\bgenial\b', 'ignorar_negativa'),
            (r'\bno entiendo\b|\bclarifica\b|\bexplica mejor\b', r'\bcomo dijiste\b|\bcomo sabes\b', 'asumir_entendimiento')
        ]
        for user_pattern, response_pattern, pattern_name in misunderstanding_patterns:
            if re.search(user_pattern, user_message.lower()):
                if not response_pattern or re.search(response_pattern, ai_response.lower()):
                    return pattern_name
        return None
    
    def update_thresholds(self, user_feedback=None):
        """Actualiza umbrales basado en patrones y feedback"""
        avg_metrics = {k: sum(v[-10:]) / max(1, len(v[-10:])) for k, v in self.evaluation_metrics.items() if v and isinstance(v[0], (int, float))}
        
        if avg_metrics.get('relevance', 0):
            self.improvement_thresholds['relevance_threshold'] = min(0.85, max(0.5, self.improvement_thresholds['relevance_threshold'] + (0.02 if avg_metrics['relevance'] > 0.8 else -0.02 if avg_metrics['relevance'] < 0.6 else 0)))
        
        if avg_metrics.get('coherence', 0):
            self.improvement_thresholds['coherence_threshold'] = min(0.9, max(0.6, self.improvement_thresholds['coherence_threshold'] + (0.01 if avg_metrics['coherence'] > 0.85 else -0.01 if avg_metrics['coherence'] < 0.7 else 0)))
        
        if avg_metrics.get('depth', 0):
            self.improvement_thresholds['elaboration_needed'] = min(0.75, max(0.45, self.improvement_thresholds['elaboration_needed'] + (0.02 if avg_metrics['depth'] > 0.7 else -0.02 if avg_metrics['depth'] < 0.5 else 0)))
        
        if user_feedback:
            feedback_value = user_feedback.get('value', 0)
            feedback_type = user_feedback.get('type', 'general')
            if feedback_type in ['relevance', 'coherence', 'depth'] and feedback_value < 0.5:
                key = {'relevance': 'relevance_threshold', 'coherence': 'coherence_threshold', 'depth': 'elaboration_needed'}[feedback_type]
                self.improvement_thresholds[key] = max(0.4, self.improvement_thresholds[key] - 0.05)
        
        self.save_metacognition()
    
    def get_performance_summary(self):
        """Genera un resumen de rendimiento"""
        avg_metrics = {k: sum(v[-10:]) / max(1, len(v[-10:])) for k, v in self.evaluation_metrics.items() if v and isinstance(v[0], (int, float))}
        return {
            'average_metrics': avg_metrics,
            'current_thresholds': self.improvement_thresholds,
            'dialog_patterns': self.dialog_patterns,
            'recent_misunderstandings': self.evaluation_metrics.get('misunderstandings', [])[-5:]
        }
    
    def reset_dialog_patterns(self):
        """Reinicia contadores de patrones problemáticos"""
        for key in self.dialog_patterns:
            self.dialog_patterns[key] = 0
        self.save_metacognition()
    
    def adjust_metrics_weights(self, user_preferences=None):
        """Ajusta los pesos de las métricas"""
        weights = {'relevance': 0.25, 'coherence': 0.20, 'helpfulness': 0.30, 'depth': 0.15, 'engagement': 0.10}
        if user_preferences:
            for metric, pref in user_preferences.items():
                if metric in weights:
                    weights[metric] = max(0.05, min(0.5, pref / 10 * 0.5))
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
        
        if self.dialog_patterns['excessive_complexity'] > 3:
            weights['depth'] = max(0.05, weights['depth'] - 0.05)
            weights['coherence'] = min(0.35, weights['coherence'] + 0.05)
            self.dialog_patterns['excessive_complexity'] -= 1
        
        if self.dialog_patterns['excessive_simplicity'] > 3:
            weights['depth'] = min(0.35, weights['depth'] + 0.05)
            self.dialog_patterns['excessive_simplicity'] -= 1

        if self.dialog_patterns['repetitive_responses'] > 1:  # Más sensible
            improvements['suggested_prompt_additions'].append(
            "Varía tu lenguaje y evita repetir frases de respuestas recientes."
            )
            improvements['adjusted_temperature'] = min(0.8, temperature + 0.1)  # Aumentar creatividad
            self.dialog_patterns['repetitive_responses'] -= 1    
        
        return weights
    
    def analyze_user_preferences(self, conversation_history):
        """Analiza preferencias del usuario"""
        if not conversation_history:
            return {}
        
        preferences = {'prefers_detailed': 0, 'prefers_examples': 0, 'prefers_technical': 0, 'prefers_informal': 0}
        indicators = {
            'prefers_detailed': ['detalle', 'explica', 'elabora', 'profundiza', 'más información'],
            'prefers_examples': ['ejemplo', 'muestra', 'ilustra', 'caso'],
            'prefers_technical': ['técnico', 'específico', 'preciso', 'académico'],
            'prefers_informal': ['simple', 'sencillo', 'fácil', 'informal', 'coloquial']
        }
        
        for message in conversation_history:
            if 'Usuario:' in message:
                user_text = message.split('Usuario: ')[1].lower()
                for pref, inds in indicators.items():
                    if any(ind in user_text for ind in inds):
                        preferences[pref] += 1
                if any(term in user_text for term in ['breve', 'corto', 'resumen', 'sintetiza']):
                    preferences['prefers_detailed'] -= 1
                if any(term in user_text for term in ['detallado', 'extenso', 'largo', 'completo']):
                    preferences['prefers_detailed'] += 1
        
        return preferences
    
    def export_learning(self):
        """Exporta aprendizajes clave"""
        avg_metrics = {k: sum(v[-20:]) / max(1, len(v[-20:])) for k, v in self.evaluation_metrics.items() if v and isinstance(v[0], (int, float))}
        return {
            'user_id': self.user_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'performance_metrics': avg_metrics,
            'problematic_patterns': {k: v for k, v in self.dialog_patterns.items() if v > 2},
            'misunderstandings': self.evaluation_metrics.get('misunderstandings', [])[-10:],
            'current_thresholds': self.improvement_thresholds
        }
    
    def aggregate_feedback(self, explicit_feedback):
        """Agrega feedback explícito"""
        if not explicit_feedback:
            return
        
        feedback_map = {'muy_util': 1.0, 'util': 0.8, 'neutral': 0.5, 'poco_util': 0.3, 'no_util': 0.1}
        feedback_type = explicit_feedback.get('type', 'general')
        feedback_value = feedback_map.get(explicit_feedback.get('value', 'neutral'), 0.5)
        
        metrics = {'relevance': 'relevance', 'helpfulness': 'helpfulness', 'clarity': 'coherence', 'depth': 'depth', 'engagement': 'engagement'}
        if feedback_type in metrics:
            self.evaluation_metrics[metrics[feedback_type]].append(feedback_value)
        else:
            for metric in ['relevance', 'coherence', 'helpfulness', 'engagement']:
                self.evaluation_metrics[metric].append(feedback_value)
        
        for metric in self.evaluation_metrics:
            if isinstance(self.evaluation_metrics[metric], list) and len(self.evaluation_metrics[metric]) > 30:
                self.evaluation_metrics[metric] = self.evaluation_metrics[metric][-30:]
        
        self.save_metacognition()
    
    def adaptive_learning(self, conversation_history, recent_evaluations):
        """Implementa aprendizaje adaptativo"""
        if not conversation_history or not recent_evaluations:
            return {}
        
        temporal_patterns = {}
        metrics = ['relevance', 'coherence', 'helpfulness', 'depth', 'engagement']
        for metric in metrics:
            if len(recent_evaluations) >= 5 and all(metric in eval_data for eval_data in recent_evaluations):
                values = [eval_data[metric] for eval_data in recent_evaluations[-5:]]
                diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
                avg_diff = sum(diffs) / len(diffs)
                temporal_patterns[f'{metric}_trend'] = avg_diff
        
        success_patterns = []
        for i, eval_data in enumerate(recent_evaluations):
            if i > 0 and all(eval_data.get(m, 0) > 0.7 for m in ['relevance', 'helpfulness']):
                msg_idx = min(i, len(conversation_history)-1)
                if 'ConsciencIA:' in conversation_history[msg_idx]:
                    success_response = conversation_history[msg_idx].split('ConsciencIA: ')[1]
                    success_patterns.append({
                        'has_examples': any(marker in success_response.lower() for marker in ['por ejemplo', 'ejemplo', 'como']),
                        'has_structure': any(marker in success_response for marker in [':', '-', '•', '*', '1.', '2.']),
                        'response_length': len(success_response.split())
                    })
        
        consolidated = {}
        if success_patterns:
            consolidated['use_examples'] = sum(1 for p in success_patterns if p['has_examples']) / len(success_patterns) > 0.6
            consolidated['use_structure'] = sum(1 for p in success_patterns if p['has_structure']) / len(success_patterns) > 0.6
            consolidated['avg_length'] = sum(p['response_length'] for p in success_patterns) / len(success_patterns)
        
        adaptive_adjustments = {}
        if consolidated.get('use_examples'): adaptive_adjustments['encourage_examples'] = True
        if consolidated.get('use_structure'): adaptive_adjustments['encourage_structure'] = True
        if 'avg_length' in consolidated: adaptive_adjustments['target_length'] = consolidated['avg_length']
        for metric, trend in temporal_patterns.items():
            if abs(trend) > 0.05 and trend < 0:
                adaptive_adjustments[f'focus_on_{metric.split("_")[0]}'] = True
        
        return adaptive_adjustments
    
    def detect_cognitive_biases(self, user_message, ai_response):
        """Detecta posibles sesgos cognitivos"""
        biases = []
        bias_patterns = [
            (r'(siempre|nunca|todos|ninguno)', 'absolutista', 'confirmation_bias'),
            (r'(como (experto|autoridad)|según todas las autoridades)', 'autoridad', 'authority_bias'),
            (r'(todos|siempre|nunca|nadie)', 'generalización', 'overgeneralization'),
            (r'(casos conocidos|ejemplos famosos)', 'disponibilidad', 'availability_bias')
        ]
        
        for pattern, context, bias_type in bias_patterns:
            if re.search(pattern, ai_response.lower()):
                biases.append({'type': bias_type, 'context': context, 'severity': 'medium'})
        
        polarity_words = {
            'positive': ['excelente', 'perfecto', 'maravilloso', 'increíble', 'fantástico'],
            'negative': ['terrible', 'horrible', 'pésimo', 'catastrófico', 'desastroso']
        }
        positive_count = sum(1 for word in polarity_words['positive'] if word in ai_response.lower())
        negative_count = sum(1 for word in polarity_words['negative'] if word in ai_response.lower())
        
        if positive_count > 3 and positive_count > negative_count * 3:
            biases.append({'type': 'positivity_bias', 'context': 'exceso de positividad', 'severity': 'medium'})
        elif negative_count > 3 and negative_count > positive_count * 3:
            biases.append({'type': 'negativity_bias', 'context': 'exceso de negatividad', 'severity': 'medium'})
        
        return biases
    
    def generate_metacognitive_report(self):
        """Genera un informe de metacognición"""
        avg_metrics = {k: sum(v[-10:]) / max(1, len(v[-10:])) for k, v in self.evaluation_metrics.items() if v and isinstance(v[0], (int, float))}
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'average_metrics': avg_metrics,
            'thresholds': self.improvement_thresholds,
            'dialog_patterns': self.dialog_patterns,
            'performance_summary': {'strengths': [], 'areas_for_improvement': [], 'recommendations': []}
        }
        
        for metric, value in avg_metrics.items():
            if value > 0.8:
                report['performance_summary']['strengths'].append(f'Alto nivel de {metric}')
            elif value < 0.6:
                report['performance_summary']['areas_for_improvement'].append(f'Mejorar {metric}')
        
        for pattern, count in self.dialog_patterns.items():
            if count > 2:
                recommendations = {
                    'repetitive_responses': 'Aumentar variedad en respuestas',
                    'topic_misalignments': 'Mejorar alineación temática',
                    'question_avoidance': 'Responder directamente a preguntas'
                }
                if pattern in recommendations:
                    report['performance_summary']['recommendations'].append(recommendations[pattern])
        
        return report
    

    def predict_user_needs(self, user_message, conversation_history, user_emotion, current_time):
        """Predice necesidades del usuario basadas en contexto multimodal"""
        needs = {
            'clarification': 0.0,
            'emotional_support': 0.0,
            'information': 0.0,
            'action_suggestion': 0.0
        }
        
        # Análisis textual
        clarification_markers = ['no entiendo', 'qué quieres decir', 'explica', 'aclara']
        if any(marker in user_message.lower() for marker in clarification_markers):
            needs['clarification'] += 0.8
        
        info_markers = ['cómo', 'qué es', 'dime', 'saber']
        if any(marker in user_message.lower() for marker in info_markers):
            needs['information'] += 0.7
        
        action_markers = ['qué hago', 'cómo puedo', 'sugerencia', 'ayuda']
        if any(marker in user_message.lower() for marker in action_markers):
            needs['action_suggestion'] += 0.6
        
        # Análisis emocional
        if user_emotion == 'negative':
            needs['emotional_support'] += 0.9
        elif user_emotion == 'positive':
            needs['emotional_support'] += 0.3
        
        # Contexto temporal
        hour = current_time.hour
        if 22 <= hour or hour < 6:  # Noche
            needs['emotional_support'] += 0.2
            needs['action_suggestion'] += 0.1
        
        # Ajuste por historial
        if len(conversation_history) > 5 and 'ConsciencIA:' in conversation_history[-1]:
            last_response = conversation_history[-1].split('ConsciencIA: ')[1]
            if '?' in user_message and not '?' in last_response:
                needs['clarification'] += 0.4
        
        primary_need = max(needs, key=needs.get)
        return {
            'primary_need': primary_need,
            'confidence': needs[primary_need],
            'all_needs': needs
        }
    
    # Clase para predecir intenciones del usuario
class IntentionPredictor:
    def __init__(self):
        # Patrones de intención basados en palabras clave y expresiones comunes
        self.intention_patterns = {
            'informative': [r'\bqué\b.*\bes\b', r'\bcómo\b.*\bfunciona\b', r'\bdime\b.*\bsobre\b',
                           r'\bexplica\b', r'\binformación\b', r'\bsaber\b.*\bacer\b'],
            'emotional': [r'\btriste\b', r'\benojado\b', r'\bfeliz\b', r'\bsiento\b', r'\bemocion\b',
                         r'\bme siento\b', r'\bnecesito\b.*\bhablar\b'],
            'help': [r'\bayuda\b', r'\bnecesito\b.*\bapoyo\b', r'\bcómo puedo\b', r'\bsolución\b',
                    r'\bproblema\b', r'\baguda\b'],
            'social': [r'\bhola\b', r'\bcuál\b.*\bes\b', r'\bhablar\b', r'\bconversar\b',
                      r'\bqué tal\b', r'\bte gusta\b'],
            'closing': [r'\badi[óo]s\b', r'\bbye\b', r'\bhasta luego\b', r'\bterminar\b',
                       r'\bgracias\b.*\bpor\b', r'\beso es todo\b']
        }
        # Pesos para cada intención (ajustables según prioridad)
        self.intention_weights = {
            'informative': 0.4,
            'emotional': 0.3,
            'help': 0.25,
            'social': 0.2,
            'closing': 0.15
        }

    def predict_intent(self, text):
        """
        Predice la intención principal del usuario basado en patrones y confianza.
        Retorna (intención, confianza) donde confianza está entre 0 y 1.
        """
        text = text.lower().strip()
        if not text:
            return 'unknown', 0.0

        scores = {'unknown': 0.0}
        for intent, patterns in self.intention_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, text))
            scores[intent] = score * self.intention_weights.get(intent, 0.1)

        # Normalizar puntajes por longitud del texto para evitar sesgos
        word_count = len(text.split())
        if word_count > 0:
            for intent in scores:
                scores[intent] /= max(1, word_count / 5)  # Normalización ligera

        total_score = sum(scores.values())
        if total_score == 0:
            return 'unknown', 0.0

        # Seleccionar la intención con mayor puntaje
        predicted_intent = max(scores, key=scores.get)
        confidence = scores[predicted_intent] / total_score if total_score > 0 else 0.0
        return predicted_intent, min(confidence, 1.0)

    def refine_intent_with_context(self, text, chat_history):
        """
        Refina la predicción usando el historial reciente para mayor precisión.
        """
        initial_intent, initial_confidence = self.predict_intent(text)
        if not chat_history or initial_confidence > 0.9:
            return initial_intent, initial_confidence

        # Analizar el historial para ajustar la intención
        recent_messages = chat_history[-min(5, len(chat_history)):]
        context_score = {intent: 0.0 for intent in self.intention_patterns}
        for msg in recent_messages:
            if "Usuario:" in msg:
                msg_text = msg.split("Usuario: ")[1]
                for intent, patterns in self.intention_patterns.items():
                    context_score[intent] += sum(1 for pattern in patterns if re.search(pattern, msg_text.lower()))

        # Combinar con la predicción inicial (70% inicial, 30% contexto)
        final_scores = {}
        for intent in self.intention_patterns:
            final_scores[intent] = (0.7 * (self.predict_intent(text)[1] if intent == initial_intent else 0)) + \
                                  (0.3 * context_score[intent] / max(1, len(recent_messages)))

        total_final = sum(final_scores.values())
        if total_final == 0:
            return initial_intent, initial_confidence

        final_intent = max(final_scores, key=final_scores.get)
        final_confidence = final_scores[final_intent] / total_final
        return final_intent, min(final_confidence, 1.0)

class DespedidaDetector:
    def __init__(self, umbral_confianza=0.75):
        self.umbral_confianza = umbral_confianza
        self.patrones = [r"\b(adi[óo]s|chao|hasta|nos vemos|me voy|gracias|chau)\b"]
    def es_despedida(self, texto, historial, user_hour):
        texto = texto.lower()
        match_count = sum(1 for patron in self.patrones if re.search(patron, texto))
        confianza = min(match_count * 0.5 + len(historial) * 0.02, 1.0)
        return confianza >= self.umbral_confianza, confianza, {}
    def generar_despedida(self, texto_usuario, personality, user_hour):
        despedidas = [f"¡Chau, {user_name}! ¡Volvé cuando quieras!", "¡Hasta pronto, amigo! Fue un gusto."]
        return random.choice(despedidas)    


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
            session.permanent = True
            login_user(user, remember=True)
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

        try:
            birth_date = datetime.datetime.strptime(birth_date_str, '%d/%m/%Y')
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
            push_enabled=False,
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
    # Inicialización básica
    session_vars = ['session_count', 'message_count', 'conversation_depth', 'emotional_state', 'user_specified_hour']
    for var in session_vars:
        if var not in session:
            if var == 'conversation_depth':
                session[var] = 0
            elif var == 'emotional_state':
                session[var] = 'neutral'
            elif var == 'user_specified_hour':
                session[var] = None
            else:
                session[var] = 0 if var.endswith('_count') else None

    if request.method == 'GET':
        session['session_count'] += 1
        session['message_count'] = 0
        session['conversation_depth'] = 0

    identifier = current_user.id
    is_premium = current_user.is_premium
    is_admin = current_user.is_admin
    user_memory = UserMemoryManager(current_user.id)
    personality_engine = PersonalityEngine(current_user.id)
    meta_cognition = MetaCognitionSystem(current_user.id)
    FREE_INTERACTION_LIMIT = 5
    interaction_count = Interaction.get_interaction_count(identifier, is_authenticated=True)
    remaining_interactions = 999999 if (is_premium or is_admin) else max(0, FREE_INTERACTION_LIMIT - interaction_count)

    # Obtener o verificar el nombre del usuario
    user_name = user_memory.get_user_name()
    user = User.query.get(current_user.id)
    if not user_name and hasattr(user, 'name') and user.name:
        user_name = user.name
        user_memory.set_user_name(user_name)
    print(f"Nombre inicial: {user_name}, Interacciones previas: {len(user_memory.memories['interactions'])}")

    # Preguntar el nombre si no existe (GET)
    if request.method == 'GET':
        if not user_name:
            response_text = "¡Hola! No nos conocemos aún, ¿cómo te llamás? Me encantaría charlar contigo."
            print("Preguntando nombre en GET")
            return render_template('consciencia.html',
                                  initial_response=response_text,
                                  remaining_interactions=remaining_interactions,
                                  is_premium=is_premium,
                                  is_admin=is_admin,
                                  is_production=os.getenv('FLASK_ENV') == 'production')

    if request.method == 'POST':
        message = request.json.get('message', '').strip()
        message_lower = message.lower()
        print(f"Mensaje recibido: {message}")

        # Detectar y guardar el nombre si el usuario lo proporciona
        if "me llamo" in message_lower or "mi nombre es" in message_lower:
            name_match = re.search(r"(me llamo|mi nombre es)\s+([a-zA-Z]+)", message_lower)
            if name_match:
                user_name = name_match.group(2).capitalize()
                user_memory.set_user_name(user_name)
                response_text = f"¡Genial, {user_name}! Ahora que nos conocemos, ¿qué tal tu día?"
                user_memory.add_interaction(message, context={'emotion': 'neutral'})
                user_memory.add_interaction(f"ConsciencIA: {response_text}", context={'emotion': 'neutral'})
                Interaction.increment_interaction(identifier, is_authenticated=True)
                print(f"Nombre guardado: {user_name}")
                return jsonify({
                    'response': response_text,
                    'remaining_interactions': remaining_interactions,
                    'limit_reached': False
                })

        # Si no hay nombre aún, insistir
        if not user_name:
            response_text = "¡Hola! Todavía no sé cómo te llamás, ¿me lo decís? ¡Así charlamos más a gusto!"
            user_memory.add_interaction(message, context={'emotion': 'neutral'})
            user_memory.add_interaction(f"ConsciencIA: {response_text}", context={'emotion': 'neutral'})
            Interaction.increment_interaction(identifier, is_authenticated=True)
            print("Insistiendo en nombre")
            return jsonify({
                'response': response_text,
                'remaining_interactions': remaining_interactions,
                'limit_reached': False
            })

        # Procesamiento normal con nombre
        session['emotional_state'] = analyze_sentiment(message)
        user_tz = timezone(timedelta(hours=user.timezone_offset if hasattr(user, 'timezone_offset') else -3))
        current_time = datetime.now(tz=user_tz)
        
        system_context = {
            "role": "system",
            "content": (
                f"ConsciencIA: alegre, relajado, empático. "
                f"Usuario: {user_name}, {session['emotional_state']}, {current_time.strftime('%H:%M')}. "
                f"Responde natural, 2-3 oraciones (60-80 tokens). "
                "No inventes nombres; si no sabés el nombre, decí 'no lo sé'."
            )
        }
        relevant_memories = user_memory.find_relevant_memories(message, limit=2)
        if relevant_memories:
            memory_snippet = "\n".join([f"- {mem['text'][:40]}... ({mem['timestamp'].strftime('%d/%m')})" for mem in relevant_memories])
            system_context["content"] += f" Recuerdos: {memory_snippet}"

        personality_instructions = personality_engine.generate_instruction_set(
            conversation_depth=session['conversation_depth'],
            context={'emotion': session['emotional_state']}
        )
        system_context["content"] += f"\n{personality_instructions['instruction_text']}"

        payload = {
            "messages": [system_context, {"role": "user", "content": message}],
            "model": "grok-2-latest",
            "stream": False,
            "temperature": personality_instructions['parameters']['temperature'],
            "max_tokens": personality_instructions['parameters']['max_tokens'],
            "top_p": 0.95
        }
        response = requests.post("https://api.x.ai/v1/chat/completions", headers={"Authorization": f"Bearer {grok_api_key}", "Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        response_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "Ups, algo salió mal.")
        print(f"Respuesta generada: {response_text}")

        user_memory.add_interaction(message, context={'emotion': session['emotional_state']})
        user_memory.add_interaction(f"ConsciencIA: {response_text}", context={'emotion': 'neutral'})
        Interaction.increment_interaction(identifier, is_authenticated=True)
        remaining_interactions = 999999 if (is_premium or is_admin) else max(0, FREE_INTERACTION_LIMIT - Interaction.get_interaction_count(identifier, is_authenticated=True))

        return jsonify({
            'response': response_text.strip(),
            'remaining_interactions': remaining_interactions,
            'limit_reached': False
        })

    # GET normal si ya hay nombre
    return render_template('consciencia.html',
                          remaining_interactions=remaining_interactions,
                          is_premium=is_premium,
                          is_admin=is_admin,
                          is_production=os.getenv('FLASK_ENV') == 'production')
                              

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
        debug=True
    )