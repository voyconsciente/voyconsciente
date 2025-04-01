import json
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, make_response, session, Response, send_file
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
from io import BytesIO
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
from typing import List, Tuple, Optional, Dict
import requests
import pdfplumber
import re
import bcrypt
import pytz
import torch  # Solo si usas PyTorch en UserMemoryManager

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
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash)

    def generate_activation_token(self):
        return serializer.dumps(self.email, salt='activation-salt')

class Reflexion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    titulo = db.Column(db.String(200), nullable=False)
    contenido = db.Column(db.Text, nullable=False)
    fecha = db.Column(db.Date, nullable=True)
    categoria = db.Column(db.String(50), nullable=True)
    imagen = db.Column(db.String(255), nullable=True)

class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    session_id = db.Column(db.String(100), nullable=True)
    interaction_date = db.Column(db.Date, nullable=False)
    interaction_count = db.Column(db.Integer, default=0)

    @classmethod
    def get_or_create(cls, identifier, is_authenticated):
        today = datetime.now(tz=timezone(timedelta(hours=-3))).date()  # UTC-3
        query = cls.query.filter_by(interaction_date=today)
        if is_authenticated:
            interaction = query.get(identifier)
        else:
            interaction = query.filter_by(session_id=identifier).first()
        if not interaction:
            interaction = cls(
                user_id=identifier if is_authenticated else None,
                session_id=identifier if not is_authenticated else None,
                interaction_date=today,
                interaction_count=0
            )
            db.session.add(interaction)
        return interaction

    @classmethod
    def increment(cls, identifier, is_authenticated):
        interaction = cls.query.filter_by(
            interaction_date=datetime.now(tz=timezone(timedelta(hours=-3))).date(),
            user_id=identifier if is_authenticated else None,
            session_id=identifier if not is_authenticated else None
        ).first()
        if interaction is None:
            interaction = cls(
                user_id=identifier if is_authenticated else None,
                session_id=identifier if not is_authenticated else None,
                interaction_date=datetime.now(tz=timezone(timedelta(hours=-3))).date(),
                interaction_count=1
            )
            db.session.add(interaction)
        else:
            interaction.interaction_count += 1
        db.session.commit()
        return interaction.interaction_count

    @classmethod
    def get_interaction_count(cls, identifier, is_authenticated):
        """Obtiene el conteo de interacciones para un identificador dado en el día actual."""
        today = datetime.now(tz=timezone(timedelta(hours=-3))).date()  # UTC-3
        interaction = cls.query.filter_by(
            interaction_date=today,
            user_id=identifier if is_authenticated else None,
            session_id=identifier if not is_authenticated else None
        ).first()
        if interaction is None:
            return 0  # Si no existe interacción, devolvemos 0
        return interaction.interaction_count

class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    file_name = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)

    __repr__ = lambda self: f"<Book {self.title}>"

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
        self.model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'rb') as f:
                self.memories = pickle.load(f)
        else:
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
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        else:
            self.index = faiss.IndexFlatL2(self.vector_dim)
        if os.path.exists(self.cluster_file):
            with open(self.cluster_file, 'rb') as f:
                self.cluster_model = pickle.load(f)

    def load_memory(self):
        """Carga memoria y asegura que los timestamps sean aware"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    self.memories = pickle.load(f)
                    for interaction in self.memories['interactions']:
                        if 'timestamp' in interaction and isinstance(interaction['timestamp'], datetime):
                            interaction['timestamp'] = interaction['timestamp'].astimezone(timezone(timedelta(hours=-3)))
                    if self.memories['last_session'] and isinstance(self.memories['last_session'], datetime):
                        self.memories['last_session'] = self.memories['last_session'].astimezone(timezone(timedelta(hours=-3)))
            else:
                self.memories = {
                    'interactions': [],
                    'topics': {},
                    'preferences': {},
                    'episodic_memories': [],
                    'last_session': None,
                    'clusters': {}
                }
        except Exception as e:
            print(f"Error al cargar memoria: {e}")
            self.memories = {
                'interactions': [],
                'topics': {},
                'preferences': {},
                'episodic_memories': [],
                'last_session': None,
                'clusters': {}
            }

    def save_memory(self):
        """Guarda memoria y clusters en disco"""
        try:
            os.makedirs("user_memories", exist_ok=True)
            with open(self.memory_file, 'wb', buffering=1024*1024) as f:
                pickle.dump(self.memories, f, protocol=pickle.HIGHEST_PROTOCOL)
            faiss.write_index(self.index, self.index_file, True)
            if self.cluster_model:
                with open(self.cluster_file, 'wb', buffering=1024*1024) as f:
                    pickle.dump(self.cluster_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Error al guardar memoria: {e}")

    def _extract_entities(self, text):
        """Extrae entidades simples (nombres, lugares, fechas)"""
        entities = {}
        try:
            matches = re.finditer(r'\b\d{1,2}/\d{1,2}/\d{4}\b', text)
            dates = [match.group(0) for match in matches]
            if dates:
                entities['dates'] = dates
        except re.error:
            pass
        return entities

    def _is_narrative_event(self, text, emotion):
        """Determina si una interacción es un evento narrativo significativo"""
        return emotion in {"positive", "negative"} and len(text.split()) > 10

    def _update_attention_weights(self, text):
        """Actualiza pesos con contexto semántico"""
        query_embedding = self.model.encode([text])[0]
        D, I = self.index.search(np.array([query_embedding], dtype=np.float32), k=10)
        now = datetime.now(tz=timezone(timedelta(hours=-3)))
        for idx, distance in zip(I[0], D[0]):
            if idx != -1 and idx < len(self.memories["interactions"]):
                mem = self.memories["interactions"][idx]
                age_factor = 1 / (1 + (now - mem["timestamp"]).total_seconds() / 86400)
                emotion_factor = {"positive": 1.3, "neutral": 1.0, "negative": 1.6}.get(mem["emotion"], 1.0)
                context_factor = 1.5 if any(k in mem["context"] for k in mem["context"]) else 1.0
                self.attention_weights[idx] = (1 - distance) * age_factor * emotion_factor * context_factor

    def _update_clusters(self):
        """Actualiza clusters basados en interacciones recientes"""
        if len(self.memories['interactions']) < 10:
            return
        embeddings = np.array([self.index.reconstruct(i['embedding_id']) for i in self.memories['interactions'][-50:]])
        from sklearn.cluster import KMeans
        self.cluster_model = KMeans(n_clusters=5, init=embeddings[:5], random_state=42, n_init=1, max_iter=10)
        self.cluster_model.fit(embeddings)
        self.memories['clusters'] = {i: int(self.cluster_model.labels_[i]) for i in range(len(self.memories['interactions'][-50:]))}

    def add_interaction(self, text, context=None, emotion=None):
        """Agrega interacción con análisis narrativo"""
        embedding = self.model.encode([text], show_progress_bar=False)[0]
        idx = self.index.ntotal
        self.index.add(np.array([embedding], dtype=np.float32))
        
        interaction = {
            'text': text,
            'timestamp': datetime.now(tz=timezone(timedelta(hours=-3))),
            'embedding_id': idx,
            'context': context or {},
            'emotion': emotion or 'neutral',
            'entities': self._extract_entities(text)
        }
        self.memories['interactions'].append(interaction)
        self.short_term_buffer.append(interaction)
        if len(self.short_term_buffer) > 15:
            self.short_term_buffer.pop(0)
        
        if self._is_narrative_event(text, emotion):
            self.memories['episodic_memories'].append(interaction)
            if len(self.memories['episodic_memories']) > 100:
                self.memories['episodic_memories'] = self.memories['episodic_memories'][-100:]
        
        self._update_attention_weights(text)
        self._update_clusters()
        if len(self.memories['interactions']) % 100 == 0:
            self.save_memory()

    def find_relevant_memories(self, query, limit=10):
        """Encuentra las interacciones más relevantes basadas en el mensaje actual"""
        if not self.memories['interactions'] or self.index.ntotal == 0:
            return []

        # Codificar el mensaje actual
        query_embedding = self.model.encode([query])[0]

        # Buscar las k interacciones más cercanas en el índice FAISS
        D, I = self.index.search(np.array([query_embedding], dtype=np.float32), k=min(limit, self.index.ntotal))

        # Obtener las interacciones correspondientes
        relevant = [
            {**self.memories['interactions'][idx], 'similarity': 1 - distance}
            for idx, distance in zip(I[0], D[0])
            if idx != -1
        ]

        return relevant

    def update_user_topics(self, topics):
        """Actualiza temas con clustering, asegurando estructura correcta"""
        now = datetime.now(tz=timezone(timedelta(hours=-3)))
        for topic in topics:
            if topic not in self.memories['topics']:
                self.memories['topics'][topic] = {'count': 0, 'first_seen': now, 'last_seen': now, 'related_clusters': []}
            self.memories['topics'][topic]['count'] += 1
            self.memories['topics'][topic]['last_seen'] = now
        self.save_memory()

    def update_preference(self, key, value):
        """Actualiza preferencias"""
        self.memories['preferences'][key] = {
            'value': value,
            'updated_at': datetime.now(tz=timezone(timedelta(hours=-3)))
        }
        if key not in self.memories['preferences'] or self.memories['preferences'][key]['value'] != value:
            self.save_memory()

    def get_memory_summary(self):
        """Resumen avanzado con narrativa, manejando datos corruptos"""
        if not self.memories['interactions']:
            return "No hay interacciones previas."
        days_since_last = (datetime.now(tz=timezone(timedelta(hours=-3))) - self.memories.get('last_session', datetime.now(tz=timezone(timedelta(hours=-3))))).days

        valid_topics = {topic: data for topic, data in self.memories.get('topics', {}).items() if isinstance(data, dict) and 'count' in data}
    
# Nueva clase PersonalityEngine
class PersonalityEngine:
    """Motor de personalidad avanzado con aprendizaje probabilístico"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.personality_file = f"user_personalities/{user_id}_personality.json"
        
        # Inicializar atributos antes de cargar datos
        self.dimensions = {}  # Inicializamos como diccionario vacío
        self._adaptive_params = {}  # Inicializamos como diccionario vacío
        
        # Cargar datos de personalidad y adaptativos
        self.load_personality()
    
    def load_personality(self):
        """Carga configuración de personalidad y adapta datos antiguos"""
        if os.path.exists(self.personality_file):
            with open(self.personality_file, 'r') as f:
                data = json.load(f)
                self.dimensions.update({
                    dim: {'mean': val, 'std': 0.1} if isinstance(val, (int, float)) else val
                    for dim, val in data.get('dimensions', {}).items()
                })
                self._adaptive_params.update(
                    {'learning_rate': 0.05, **data.get('adaptive_params', {})}
                )

# Nueva clase MetaCognitionSystem
class MetaCognitionSystem:
    """Sistema de meta-cognición para permitir que ConciencIA evalúe y mejore sus respuestas"""
    
    def __init__(self, user_id, custom_thresholds=None, custom_weights=None):
        self.user_id = user_id
        self.meta_file = f"user_metacognition/{user_id}_metacog.json"
        self._changes_since_save = 0
        
        # Cargar datos existentes
        self.load_metacognition()
        
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
    
    def load_metacognition(self):
        """Carga datos de meta-cognición si existen"""
        try:
            if os.path.exists(self.meta_file):
                with open(self.meta_file, 'rb') as f:
                    self.evaluation_metrics, self.improvement_thresholds, self.dialog_patterns = pickle.load(f)
        except Exception as e:
            print(f"Error al cargar meta-cognición: {e}")
    
    def save_metacognition(self):
        """Guarda datos de meta-cognición con debounce"""
        self._changes_since_save += 1
        if self._changes_since_save >= 5:  # Guardar cada 5 cambios
            try:
                with open(self.meta_file, 'wb') as f:
                    pickle.dump((self.evaluation_metrics, self.improvement_thresholds, self.dialog_patterns), f, protocol=pickle.HIGHEST_PROTOCOL)
                self._changes_since_save = 0
            except Exception as e:
                print(f"Error al guardar meta-cognición: {e}")
    
    def evaluate_response(self, user_message, ai_response, previous_messages=None, user_emotion='neutral', current_time=None):
        """Evalúa la calidad de la respuesta con predicción de necesidades"""
        if not previous_messages:
            previous_messages = []
        if not current_time:
            current_time = datetime.now(tz=timezone(timedelta(hours=-3)))

        try:
            # Predicción de necesidades
            needs_pred = self.predict_user_needs(user_message, previous_messages, user_emotion, current_time)
            primary_need = needs_pred['primary_need']

            # Extracción de palabras clave
            user_keywords = set(self._extract_keywords(user_message))
            response_keywords = set(self._extract_keywords(ai_response))
            semantic_overlap = len(user_keywords & response_keywords) / max(1, len(user_keywords))

            # Calcular métricas
            scores = {
                'relevance': min(1.0, semantic_overlap * 1.5),
                'coherence': 0.9 - (0.5 if self._detect_contradictions(ai_response) else 0.0),
                'depth': self.calculate_depth(ai_response),
                'engagement': self.calculate_engagement(ai_response),
                'helpfulness': self.calculate_helpfulness(primary_need, user_message, ai_response, semantic_overlap),
                'need_satisfaction': self.calculate_need_satisfaction(primary_need, ai_response.lower())
            }

            # Actualizar patrones y métricas
            self._update_dialog_patterns(user_message, ai_response, previous_messages)
            for metric, value in scores.items():
                self.evaluation_metrics[metric].append(value)
                if len(self.evaluation_metrics[metric]) > 20:
                    self.evaluation_metrics[metric] = self.evaluation_metrics[metric][-20:]
            self.save_metacognition()

        except Exception as e:
            print(f"Error al evaluar respuesta: {e}")
            scores = {metric: 0.3 for metric in scores}

        return scores

    def calculate_depth(self, ai_response):
        words = ai_response.lower().split()
        unique_words = len(set(words))
        long_words = sum(len(w) > 7 for w in words)
        return min(1.0, (unique_words / len(words) * 0.5) +
                   (long_words / len(words) * 2.0) +
                   (len(words) / 200 * 0.3))

    def calculate_engagement(self, ai_response):
        return min(1.0, 0.5 +
                   0.2 * ('?' in ai_response) +
                   0.15 * any(marker in ai_response.lower() for marker in {'por ejemplo', 'ejemplo', 'como cuando', 'imagina'}))

    def calculate_helpfulness(self, primary_need, user_message, ai_response, semantic_overlap):
        if primary_need == 'information' and any(marker in user_message.lower() for marker in {'cómo', 'qué', 'dime', 'saber'}):
            return 0.7 + semantic_overlap * 0.3 if semantic_overlap > 0.6 else 0.4
        elif primary_need == 'emotional_support':
            return 0.6 + 0.4 * ('apoyo' in ai_response.lower() or 'entiendo' in ai_response.lower())
        else:
            return 0.5 + semantic_overlap * 0.2 + self.calculate_engagement(ai_response) * 0.3

    def calculate_need_satisfaction(self, primary_need, response_lower):
        need_markers = {
            'clarification': {'aclaro', 'explico', 'entiendo', 'claro'},
            'emotional_support': {'ánimo', 'entiendo', 'estoy aquí', 'tranquilo'},
            'information': {'dato', 'explicación', 'saber', 'aquí tienes'},
            'action_suggestion': {'sugiero', 'puedes', 'intenta', 'prueba'}
        }
        markers_found = len(need_markers.get(primary_need, set()) & set(response_lower.split()))
        return min(1.0, 0.5 + (markers_found * 0.2))  # Base 0.5, +0.2 por marcador
    
    
    def generate_self_improvements(self, scores, user_message, ai_response, intent='unknown'):
        """Genera mejoras basadas en la evaluación de la respuesta"""
        improvements = {
            'adjusted_temperature': None,
            'suggested_prompt_additions': [],
            'focus_areas': []
        }

        # Define conditions and corresponding improvements in a structured way
        conditions = [
            (scores.get('relevance', 0) < 0.5, 
             "Asegúrate de responder directamente a lo que el usuario menciona."),
            (scores.get('coherence', 0) < 0.6, 
             "Revisa que tus ideas sean consistentes y no te contradigas."),
            (scores.get('depth', 0) < 0.4, 
             "Añade un poco más de detalle o una idea extra si encaja.", 
             "Considerá buscar info externa (web o X) para sumar un dato fresco y enriquecer la respuesta.",
             'usar_búsqueda_externa'),
            (scores.get('engagement', 0) < 0.5, 
             "Intenta enganchar más con una pregunta o un ejemplo sencillo."),
            (self.dialog_patterns['repetitive_responses'] > 1, 
             "Varía tu lenguaje y evita repetir frases o ideas de respuestas recientes.", 
             min(0.8, personality_instructions['parameters']['temperature'] + 0.1)),
            (scores.get('helpfulness', 0) > 0.7 and scores.get('engagement', 0) > 0.6, 
             "Sigue destacando un enfoque positivo y motivador; parece que está funcionando bien."),
            (scores.get('helpfulness', 0) < 0.4, 
             "Intenta añadir un toque más optimista o una idea práctica para ser más útil."),
            ('todo estará bien' in ai_response.lower() or 'seguro que' in ai_response.lower(), 
             "Evita sonar demasiado seguro o prometedor; sé más humilde y práctico."),
            (scores.get('helpfulness', 0) < 0.5 and intent in ['information', 'action_suggestion'], 
             "Da una idea más concreta y realista para que sea más útil.")
        ]

        # Apply conditions
        for condition in conditions:
            if condition[0]:
                improvements['suggested_prompt_additions'].extend(condition[1:-1])
                if len(condition) > 2 and isinstance(condition[-1], str):
                    improvements['focus_areas'].append(condition[-1])
                elif len(condition) > 2:
                    improvements['adjusted_temperature'] = condition[-1]

        # Handle explicit user instructions
        user_instructions = [
            ("no me recuerdes" in user_message.lower() or "no repitas" in user_message.lower(), 
             "No menciones recuerdos específicos del pasado a menos que el usuario lo pida explícitamente.", 
             lambda: self.dialog_patterns.update({'repetitive_responses': self.dialog_patterns['repetitive_responses'] + 1})),
            ("actúa más normal" in user_message.lower() or "sé más natural" in user_message.lower(), 
             "Adopta un tono más casual y relajado, evitando exageraciones o referencias innecesarias.", 
             lambda: self.dialog_patterns.update({'excessive_complexity': max(0, self.dialog_patterns['excessive_complexity'] - 1)}),
             0.6)
        ]

        for instruction in user_instructions:
            if instruction[0]:
                improvements['suggested_prompt_additions'].append(instruction[1])
                if callable(instruction[2]):
                    instruction[2]()
                if len(instruction) > 3:
                    improvements['adjusted_temperature'] = instruction[3]

        # Adjustments based on metrics
        avg_metrics = {k: sum(v[-5:]) / max(1, len(v[-5:])) for k, v in self.evaluation_metrics.items() if v}
        metric_adjustments = [
            (avg_metrics.get('relevance', 0.7) < self.improvement_thresholds['relevance_threshold'], 
             0.65, 
             'aumentar_relevancia'),
            (avg_metrics.get('coherence', 0.8) < self.improvement_thresholds['coherence_threshold'], 
             None, 
             'mejorar_coherencia'),
            (scores.get('depth', 0.5) < self.improvement_thresholds['elaboration_needed'], 
             150, 
             None)
        ]

        for adjustment in metric_adjustments:
            if adjustment[0]:
                if adjustment[1] is not None:
                    improvements['adjusted_temperature'] = adjustment[1]
                if adjustment[2]:
                    improvements['focus_areas'].append(adjustment[2])

        self.save_metacognition()
        print(f"Mejoras sugeridas: {improvements}")
        return improvements
    
    def _extract_keywords(self, text):
        """Extrae palabras clave de un texto"""
        try:
            stopwords = {'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'de', 'del', 'a', 'en', 'que', 'es', 'por', 'para', 'con', 'su', 'sus'}
            return [word for word in text.lower().split() if len(word) > 3 and word not in stopwords]
        except Exception as e:
            print(f"Error al extraer keywords: {e}")
            return []
    
    def _detect_contradictions(self, text):
        """Detecta posibles contradicciones internas en el texto"""
        try:
            patterns = (r'no .{1,20} pero .{1,20} sí',
                        r'siempre .{1,30} nunca',
                        r'es .{1,20} no es')
            regex = re.compile('|'.join(patterns))
            return bool(regex.search(text.lower()))
        except Exception as e:
            print(f"Error al detectar contradicciones: {e}")
            return False
    
    def _update_dialog_patterns(self, user_message, ai_response, previous_messages):
        repeated_phrases = 0
        question_avoidance = False
        topic_misalignment = False
        
        if len(previous_messages) >= 2:
            last_response = previous_messages[-1]
            repeated_phrases = self._find_repeated_phrases(last_response, ai_response) >= 1
        
        if '?' in user_message and '?' not in ai_response:
            question_words = {'cómo', 'qué', 'cuándo', 'dónde', 'por qué', 'cuál', 'quién'}
            question_avoidance = any(word in user_message.lower() for word in question_words)
        
        if previous_messages and len(previous_messages) >= 2:
            prev_user_message = previous_messages[-2]
            user_topics = set(self._extract_keywords(user_message))
            prev_topics = set(self._extract_keywords(prev_user_message))
            topic_misalignment = len(user_topics & prev_topics) > 2 and len(user_topics & set(self._extract_keywords(ai_response))) < 2
        
        if repeated_phrases:
            self.dialog_patterns['repetitive_responses'] += 1
        if question_avoidance:
            self.dialog_patterns['question_avoidance'] += 1
        if topic_misalignment:
            self.dialog_patterns['topic_misalignments'] += 1
    
    def _find_repeated_phrases(self, text1, text2, min_phrase_length=5):
        """Encuentra frases repetidas entre dos textos"""
        seen = set()
        repeated = 0
        for i in range(len(text1.split()) - min_phrase_length + 1):
            phrase = ' '.join(text1.lower().split()[i:i+min_phrase_length])
            if phrase in seen or phrase in text2.lower().split():
                repeated += 1
            seen.add(phrase)
        return repeated
    
    def _detect_misunderstanding(self, user_message, ai_response):
        """Detecta posibles malentendidos en la respuesta"""
        user_message = user_message.lower()
        ai_response = ai_response.lower()
        patterns = [
            (r'\?.*\bcómo\b|\bqué\b|\bcuál\b|\bcuándo\b', 'respuesta_general_a_pregunta_específica'),
            (r'\bno\b.*\b(quiero|me gusta|deseo)\b', r'(?:\bperfecto\b|\bexcelente\b|\bgenial\b)', 'ignorar_negativa'),
            (r'\bno entiendo\b|\bclarifica\b|\bexplica mejor\b', r'(?:\bcomo dijiste\b|\bcomo sabes\b)', 'asumir_entendimiento')
        ]
        for user_pattern, response_pattern, pattern_name in patterns:
            if re.search(user_pattern, user_message) and (not response_pattern or re.search(response_pattern, ai_response)):
                return pattern_name
        return None
    
    def update_thresholds(self, user_feedback=None):
        """Actualiza umbrales basado en patrones y feedback"""
        avg_metrics = {k: sum(v[-10:])/len(v[-10:]) if v else 0 for k, v in self.evaluation_metrics.items() if v and isinstance(v[0], (int, float))}
        
        for metric, value in avg_metrics.items():
            if metric == 'relevance':
                self.improvement_thresholds['relevance_threshold'] = min(0.85, max(0.5, self.improvement_thresholds['relevance_threshold'] + (0.02 if value > 0.8 else -0.02 if value < 0.6 else 0)))
            elif metric == 'coherence':
                self.improvement_thresholds['coherence_threshold'] = min(0.9, max(0.6, self.improvement_thresholds['coherence_threshold'] + (0.01 if value > 0.85 else -0.01 if value < 0.7 else 0)))
            elif metric == 'depth':
                self.improvement_thresholds['elaboration_needed'] = min(0.75, max(0.45, self.improvement_thresholds['elaboration_needed'] + (0.02 if value > 0.7 else -0.02 if value < 0.5 else 0)))
        
        if user_feedback:
            feedback_value = user_feedback.get('value', 0)
            feedback_type = user_feedback.get('type', 'general')
            if feedback_type in ['relevance', 'coherence', 'depth'] and feedback_value < 0.5:
                key = {'relevance': 'relevance_threshold', 'coherence': 'coherence_threshold', 'depth': 'elaboration_needed'}[feedback_type]
                self.improvement_thresholds[key] = max(0.4, self.improvement_thresholds[key] - 0.05)
        
        self.save_metacognition()
    
    def get_performance_summary(self):
        """Genera un resumen de rendimiento"""
        avg_metrics = {}
        for k, v in self.evaluation_metrics.items():
            if v and isinstance(v[0], (int, float)):
                avg_metrics[k] = sum(v[-10:]) / max(1, len(v[-10:]))
        
        return {
            'average_metrics': avg_metrics,
            'current_thresholds': self.improvement_thresholds,
            'dialog_patterns': self.dialog_patterns,
            'recent_misunderstandings': self.evaluation_metrics.get('misunderstandings', [])[-5:]
        }
    
    def reset_dialog_patterns(self):
        """Reinicia contadores de patrones problemáticos"""
        self.dialog_patterns = {k: 0 for k in self.dialog_patterns}
        self.save_metacognition()
    
    def adjust_metrics_weights(self, user_preferences=None):
        """Ajusta los pesos de las métricas"""
        weights = {'relevance': 0.25, 'coherence': 0.20, 'helpfulness': 0.30, 'depth': 0.15, 'engagement': 0.10}
        
        if user_preferences:
            weights.update({metric: max(0.05, min(0.5, pref / 10 * 0.5))
                            for metric, pref in user_preferences.items() if metric in weights})
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
        
        dialog_adjustments = {
            'excessive_complexity': (3, 'depth', -0.05, 'coherence', 0.05),
            'excessive_simplicity': (3, 'depth', 0.05)
        }

        for pattern, (threshold, *adjustments) in dialog_adjustments.items():
            if self.dialog_patterns[pattern] > threshold:
                for i in range(0, len(adjustments), 2):
                    metric, adjustment = adjustments[i], adjustments[i+1]
                    weights[metric] = max(0.05, min(0.35, weights[metric] + adjustment))
                self.dialog_patterns[pattern] -= 1

        if self.dialog_patterns['repetitive_responses'] > 1:
            improvements['suggested_prompt_additions'].append(
                "Varía tu lenguaje y evita repetir frases de respuestas recientes."
            )
            improvements['adjusted_temperature'] = min(0.8, temperature + 0.1)
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
        
        detailed_decrement_terms = {'breve', 'corto', 'resumen', 'sintetiza'}
        detailed_increment_terms = {'detallado', 'extenso', 'largo', 'completo'}
        
        for message in filter(lambda msg: 'Usuario:' in msg, conversation_history):
            user_text = message.split('Usuario: ')[1].lower()
            for pref, inds in indicators.items():
                if any(ind in user_text for ind in inds):
                    preferences[pref] += 1
            if detailed_decrement_terms.intersection(user_text.split()):
                preferences['prefers_detailed'] -= 1
            if detailed_increment_terms.intersection(user_text.split()):
                preferences['prefers_detailed'] += 1

        return preferences
    
    def export_learning(self):
        """Exporta aprendizajes clave"""
        avg_metrics = {}
        for k, v in self.evaluation_metrics.items():
            if v and isinstance(v[0], (int, float)):
                avg_metrics[k] = sum(v[-20:]) / max(1, len(v[-20:]))
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
            self.evaluation_metrics.setdefault(metrics[feedback_type], []).append(feedback_value)
        else:
            for metric in ['relevance', 'coherence', 'helpfulness', 'engagement']:
                self.evaluation_metrics.setdefault(metric, []).append(feedback_value)
        
        self._trim_metrics()
        self.save_metacognition()
    
    def _trim_metrics(self):
        for metric, values in self.evaluation_metrics.items():
            if not isinstance(values, list):
                continue
            self.evaluation_metrics[metric] = values[-30:]
    
    def adaptive_learning(self, conversation_history, recent_evaluations):
        """Implementa aprendizaje adaptativo"""
        if not conversation_history or not recent_evaluations:
            return {}
        
        recent_evals = {m: [eval_data[m] for eval_data in recent_evaluations[-5:] if m in eval_data] for m in ['relevance', 'coherence', 'helpfulness', 'depth', 'engagement']}
        temporal_patterns = {f'{m}_trend': sum(diffs) / len(diffs) if diffs else 0 for m, diffs in {m: [v[i+1] - v[i] for i in range(len(v)-1)] for m, v in recent_evals.items()}.items()}
        
        success_responses = [conversation_history[msg_idx].split('ConsciencIA: ')[1] for msg_idx, eval_data in enumerate(recent_evaluations) if msg_idx > 0 and all(eval_data.get(m, 0) > 0.7 for m in ['relevance', 'helpfulness'])]
        success_patterns = [{'has_examples': any(marker in success_response.lower() for marker in ['por ejemplo', 'ejemplo', 'como']), 'has_structure': any(marker in success_response for marker in [':', '-', '•', '*', '1.', '2.']), 'response_length': len(success_response.split())} for success_response in success_responses]
        
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
        patterns = {
            'confirmation_bias': (r'(siempre|nunca|todos|ninguno)', 'absolutista'),
            'authority_bias': (r'(como (experto|autoridad)|según todas las autoridades)', 'autoridad'),
            'overgeneralization': (r'(todos|siempre|nunca|nadie)', 'generalización'),
            'availability_bias': (r'(casos conocidos|ejemplos famosos)', 'disponibilidad')
        }
        polarity_words = {
            'positive': set(['excelente', 'perfecto', 'maravilloso', 'increíble', 'fantástico']),
            'negative': set(['terrible', 'horrible', 'pésimo', 'catastrófico', 'desastroso'])
        }
        response_words = set(re.findall(r'\w+', ai_response.lower()))
        
        for bias_type, (pattern, context) in patterns.items():
            if re.search(pattern, ai_response.lower()):
                biases.append({'type': bias_type, 'context': context, 'severity': 'medium'})
        
        positive_count = len(response_words & polarity_words['positive'])
        negative_count = len(response_words & polarity_words['negative'])
        
        if positive_count > 3 and positive_count > negative_count * 3:
            biases.append({'type': 'positivity_bias', 'context': 'exceso de positividad', 'severity': 'medium'})
        elif negative_count > 3 and negative_count > positive_count * 3:
            biases.append({'type': 'negativity_bias', 'context': 'exceso de negatividad', 'severity': 'medium'})
        
        return biases
    
    def generate_metacognitive_report(self):
        """Genera un informe de metacognición"""
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'average_metrics': {k: sum(v[-10:]) / max(1, len(v[-10:])) for k, v in self.evaluation_metrics.items() if v and isinstance(v[0], (int, float))},
            'thresholds': self.improvement_thresholds,
            'dialog_patterns': self.dialog_patterns,
            'performance_summary': {'strengths': [], 'areas_for_improvement': [], 'recommendations': []}
        }
        
        for metric, value in report['average_metrics'].items():
            if value > 0.8:
                report['performance_summary']['strengths'].append(f'Alto nivel de {metric}')
            elif value < 0.6:
                report['performance_summary']['areas_for_improvement'].append(f'Mejorar {metric}')
        
        for pattern, count in report['dialog_patterns'].items():
            if count > 2:
                if pattern in ('repetitive_responses', 'topic_misalignments', 'question_avoidance'):
                    report['performance_summary']['recommendations'].append(RECOMMENDATIONS[pattern])
        
        return report
    

    def predict_user_needs(self, user_message, conversation_history, user_emotion, current_time):
        """Predice necesidades del usuario basadas en contexto multimodal"""
        # Análisis textual
        needs = {'clarification': 0.0, 'emotional_support': 0.0, 'information': 0.0, 'action_suggestion': 0.0}
        for marker in ['no entiendo', 'qué quieres decir', 'explica', 'aclara']:
            if marker in user_message.lower():
                needs['clarification'] += 0.8
        for marker in ['cómo', 'qué es', 'dime', 'saber']:
            if marker in user_message.lower():
                needs['information'] += 0.7
        for marker in ['qué hago', 'cómo puedo', 'sugerencia', 'ayuda']:
            if marker in user_message.lower():
                needs['action_suggestion'] += 0.6
        
        # Análisis emocional
        if user_emotion == 'negative':
            needs['emotional_support'] += 0.9
        elif user_emotion == 'positive':
            needs['emotional_support'] += 0.3
        
        # Contexto temporal
        if 22 <= current_time.hour or current_time.hour < 6:  # Noche
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
        self.intention_weights = {
            'informative': 0.4,
            'emotional': 0.3,
            'help': 0.25,
            'social': 0.2,
            'closing': 0.15
        }

    def predict_intent(self, text):
        text = text.lower().strip()
        if not text:
            return 'unknown', 0.0

        scores = {intent: sum(re.search(pattern, text) is not None for pattern in patterns) 
                  * self.intention_weights.get(intent, 0.1)
                  for intent, patterns in self.intention_patterns.items()}

        word_count = len(text.split())
        normalization_factor = max(1, word_count / 5)
        scores = {intent: score / normalization_factor for intent, score in scores.items()}

        if not any(scores.values()):
            return 'unknown', 0.0

        predicted_intent, confidence = max(scores.items(), key=lambda item: item[1])
        return predicted_intent, min(confidence, 1.0)

    def refine_intent_with_context(self, text, chat_history):
        initial_intent, initial_confidence = self.predict_intent(text)
        if not chat_history or initial_confidence > 0.9:
            return initial_intent, initial_confidence

        recent_messages = [msg.split("Usuario: ")[1].lower() for msg in chat_history[-5:] if "Usuario:" in msg]
        context_score = {intent: sum(re.search(pattern, msg) is not None for msg in recent_messages for pattern in patterns) 
                         for intent, patterns in self.intention_patterns.items()}

        final_scores = {intent: 0.7 * (initial_confidence if intent == initial_intent else 0) +
                        0.3 * context_score[intent] / max(1, len(recent_messages))
                        for intent in self.intention_patterns}

        if not any(final_scores.values()):
            return initial_intent, initial_confidence

        final_intent = max(final_scores, key=final_scores.get)
        final_confidence = final_scores[final_intent] / sum(final_scores.values())
        return final_intent, min(final_confidence, 1.0)

    
import re
import random

class DespedidaDetector:
    """Sistema avanzado para detectar despedidas y generar respuestas adecuadas."""
    
    def __init__(self, umbral_confianza: float = 0.70):
        self.umbral_confianza = umbral_confianza
        self.patrones_despedida = {
            "es": [
                r"\badi[óo]s\b", r"\bchao\b", r"\bhasta luego\b", r"\bhasta pronto\b",
                r"\bnos vemos\b", r"\bhasta ma[nñ]ana\b", r"\bme voy\b", r"\bmuchas gracias\b",
                r"\bgracias por todo\b", r"\bme despido\b", r"\bbuenas noches\b",
                r"\bhasta la pr[óo]xima\b", r"\bfin\b", r"\bterminar\b", r"\bterminamos\b",
                r"\beso es todo\b", r"\bya est[áa]\b", r"\blisto\b", r"\bes todo\b"
            ],
            "en": [
                r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b", r"\blater\b", r"\bcya\b",
                r"\bfarewell\b", r"\bhave a good\b", r"\btake care\b", r"\bending\b",
                r"\bthank you for\b", r"\bthat's all\b", r"\bi'm done\b", r"\bthat will be all\b",
                r"\bgood night\b", r"\bsigning off\b"
            ]
        }
        self.respuestas_despedida = {
            "es": [
                "¡Hasta pronto! Ha sido un placer charlar contigo.",
                "Gracias por conversar. ¡Que tengas un lindo día!",
                "Entendido, me despido aquí. ¡Siempre un gusto ayudarte!",
                "¡Perfecto! Nos vemos cuando quieras. ¡Adiós!",
                "¡Hasta la próxima! Fue genial hablar contigo."
            ],
            "en": [
                "See you later! It’s been great chatting with you.",
                "Thanks for the talk. Have a wonderful day!",
                "Got it, I’ll say goodbye here. Always a pleasure to assist!",
                "Perfect! Catch you next time. Bye!",
                "Until next time! Loved our conversation."
            ]
        }


# Función para enviar reflexiones programadas
def send_weekly_reflection():
    with app.app_context():
        users = User.query.filter_by(is_active=True).all()
        reflections = Reflexion.query.all()
        random.shuffle(reflections)
        for user in users:
            categories = user.preferred_categories.split(',')
            reflection = next((r for r in reflections if any(c in categories for c in r.categoria.split(','))), None)

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
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        width, height = img.size
        if width > max_width:
            new_height = int((max_width / width) * height)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, 'JPEG', quality=quality, optimize=True)
        buf.seek(0)
        with open(output_path, 'wb') as f:
            f.write(buf.read())
        print(f"Imagen comprimida: {output_path}")
    except Exception as e:
        print(f"Error al comprimir imagen: {e}")

@app.route('/logout')
@login_required
def logout() -> Response:
    """Clears the session and redirects to the homepage.

    Returns:
        Response: A redirect response to the homepage.
    """
    session.clear()
    return redirect(url_for('home'))

@app.route('/update_push', methods=['POST'])
def update_push():
    data = request.get_json()
    email = data.get('email')
    player_id = data.get('player_id')
    if not email:
        return jsonify({'status': 'error'})

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'status': 'error'})

    if user.onesignal_player_id != player_id:
        user.onesignal_player_id = player_id
        db.session.commit()

    return jsonify({'status': 'success'})

@app.route('/terminos-condiciones')
def terminos_condiciones():
    return render_template('terminos_condiciones.html', año_actual=datetime.date.today().year)

@app.route('/politica-privacidad')
def politica_privacidad():
    current_year = datetime.datetime.now().year
    return render_template('politica_privacidad.html', año_actual=current_year)

@app.route('/analisis')
def analisis():
    metrics_data = []
    try:
        response = analytics_client.run_report({
            'property': f'properties/{GA_PROPERTY_ID}',
            'date_ranges': [{'start_date': '7daysAgo', 'end_date': 'today'}],
            'dimensions': [{'name': 'date'}],
            'metrics': [
                {'name': 'activeUsers'},
                {'name': 'sessions'},
                {'name': 'averageSessionDuration'},
                {'name': 'screenPageViews'}
            ]
        })
        for row in response.rows:
            metrics_data.append({
                'date': row.dimension_values[0].value,
                'active_users': row.metric_values[0].value,
                'sessions': row.metric_values[1].value,
                'avg_session_duration': float(row.metric_values[2].value) / 60,
                'page_views': row.metric_values[3].value
            })
    except Exception as e:
        print(f"Error al obtener métricas de Google Analytics: {e}")
        metrics_data = []
        flash('No se pudieron cargar las métricas en este momento.', 'error')
    return render_template('analisis.html',
                          metrics=metrics_data,
                          flow_name=GA_FLOW_NAME,
                          flow_url=GA_FLOW_URL,
                          flow_id=GA_FLOW_ID)

@app.route('/consciencia-info')
def consciencia_info():
    return send_from_directory('templates', 'consciencia_info.html')

@app.route('/static/js/main.js')
def serve_main_js():
    return send_from_directory(app.static_folder, 'js/main.js')

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
    return send_from_directory(app.static_folder, 'OneSignalSDKWorker.js', mimetype='application/javascript')

@app.route('/test-static/<path:filename>')
def test_static(filename):
    return send_from_directory(app.static_folder, filename, cache_timeout=3600)

@app.route('/')
@cache.cached(timeout=300)
def home():
    return render_template('index.html', año_actual=datetime.now().year)

@app.route('/sobre-nosotros')
@cache.cached(timeout=300)
def sobre_nosotros():
    return render_template('sobre_nosotros.html')

@app.route('/reflexiones', defaults={'page': 1})
@app.route('/reflexiones/page/<int:page>')
@cache.cached(timeout=300, query_string=True)
def mostrar_reflexiones(page):
    per_page = 20
    reflexiones = Reflexion.query.paginate(page=page, per_page=per_page, error_out=False)
    return render_template('reflexiones.html', reflexiones=reflexiones.items, pagination=reflexiones)

@app.route('/reflexiones/<categoria>', defaults={'page': 1})
@app.route('/reflexiones/<categoria>/page/<int:page>')
@app.route('/reflexiones/<categoria>/<int:page>')
@cache.cached(timeout=300, query_string=True)
def reflexiones_por_categoria(categoria, page=1):
    per_page = 20
    reflexiones = db.session.query(Reflexion).filter_by(categoria=categoria).order_by(Reflexion.id.desc()).paginate(page=page, per_page=per_page, error_out=False)
    print(f"Página {page}, Categoría {categoria}: {len(reflexiones.items)} reflexiones enviadas de {reflexiones.total} totales")
    return render_template('reflexiones.html', reflexiones=reflexiones.items, pagination=reflexiones, categoria=categoria)

@app.route('/reflexion/<int:id>')
@cache.cached(timeout=300, query_string=True)
def mostrar_reflexion(id):
    reflexion = Reflexion.query.get_or_404(id)
    return render_template('reflexion.html', reflexion=reflexion)

@app.route('/articulo-aleatorio')
def articulo_aleatorio():
    # We only need the ID, so query only that
    random_id = random.choice(db.session.query(Reflexion.id).all())[0]
    return redirect(url_for('mostrar_reflexion', id=random_id))

@app.route('/galeria', defaults={'page': 1})
@app.route('/galeria/page/<int:page>')
@cache.cached(timeout=300, query_string=True)
def galeria(page):
    per_page = 20
    reflexiones_pagination = Reflexion.query.paginate(page=page, per_page=per_page, error_out=False)
    reflexiones_items = reflexiones_pagination.items
    total_reflexiones = reflexiones_pagination.total
    print(f"Galería - Página {page}: {len(reflexiones_items)} reflexiones enviadas de {total_reflexiones} totales")
    return render_template('galeria.html', reflexiones=reflexiones_items, pagination=reflexiones_pagination)

@app.route('/recursos')
@cache.cached(timeout=300)
def recursos():
    return render_template('recursos.html')

@app.route('/contacto', methods=['GET', 'POST'])
def contacto():
    if request.method == 'POST':
        nombre = request.form['nombre']
        email = request.form['email']
        mensaje = request.form['mensaje']
        msg = Message(subject=f"Nuevo mensaje de {nombre}", recipients=[app.config['MAIL_USERNAME']],
                      body=f"Nombre: {nombre}\nCorreo: {email}\nMensaje: {mensaje}")
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

    # Query database only once
    resultados = Reflexion.query.filter(
        db.or_(Reflexion.titulo.ilike(f'%{query}%'), Reflexion.contenido.ilike(f'%{query}%'))
    ).count()

    # Use SQLAlchemy's built-in pagination
    paginated_resultados = Reflexion.query.filter(
        db.or_(Reflexion.titulo.ilike(f'%{query}%'), Reflexion.contenido.ilike(f'%{query}%'))
    ).paginate(page=page, per_page=per_page, error_out=False)

    return render_template('busqueda.html', resultados=paginated_resultados.items, query=query, page=page, per_page=per_page, total=resultados)

from typing import Optional, Union
from flask import Response, Request

@app.route('/login', methods=['GET', 'POST'])
def login() -> Union[Response, str]:
    """
    Handles user login.

    Returns:
        Union[Response, str]: A redirect response to the next page or home, or renders the login page.
    """
    if current_user.is_authenticated:
        next_page: str = request.args.get('next', '')
        print(f"Usuario ya autenticado, redirigiendo a: {next_page if next_page and is_safe_url(next_page) else url_for('home')}")
        if next_page and is_safe_url(next_page):
            return redirect(next_page)
        return redirect(url_for('home'))

    if request.method == 'POST':
        email: str = request.form['email']
        password: str = request.form['password']
        user: Optional[User] = User.query.filter_by(email=email).first()
        next_page: str = request.form.get('next', request.args.get('next', ''))

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

    next_page: str = request.args.get('next', '')
    email: str = request.args.get('email', '')
    print(f"Valor de next_page en GET: {next_page}")
    return render_template('login.html', next=next_page, email=email)

@app.route('/register', methods=['GET', 'POST'])
def register(next_page: str = '') -> Union[Response, str]:
    """
    Handles user registration.

    Args:
        next_page (str): The URL to redirect to after registration.

    Returns:
        Union[Response, str]: A redirect response to the next page or home, or renders the registration page.
    """
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        name = request.form['name']
        birth_date_str = request.form['birth_date']
        phone = request.form.get('phone', '')

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

    return render_template('register.html', next=next_page)

@app.route('/activate/<token>')
def activate(token: str) -> Union[Response, str]:
    """
    Activates a user account based on the provided token.

    Args:
        token (str): The activation token for the user.

    Returns:
        Union[Response, str]: A redirect response to the appropriate page.
    """
    try:
        email: str = serializer.loads(token, salt='activation-salt', max_age=3600)
        user: Optional[User] = User.query.filter_by(email=email).first()
        if not user:
            flash('El enlace de activación es inválido o el usuario no existe.', 'error')
            return redirect(url_for('register', next=request.args.get('next')))
        if user.is_active:
            flash('Tu cuenta ya está activada.', 'success')
            next_page: str = request.args.get('next', '')
            if next_page and is_safe_url(next_page):
                return redirect(next_page)
            return redirect(url_for('home'))
        user.is_active = True
        user.activation_token = None
        db.session.commit()
        flash('¡Tu cuenta ha sido activada! Por favor, inicia sesión.', 'success')
        next_page: str = request.args.get('next', '')
        return redirect(url_for('login', next=next_page))
    except (SignatureExpired, BadSignature):
        flash('El enlace de activación ha expirado o es inválido.', 'error')
        return redirect(url_for('register', next=request.args.get('next')))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password() -> Response:
    """
    Handles the forgot password page.

    If the request method is POST, it sends an email to the user with a link to reset their password.
    If the request method is GET, it renders the forgot password page.

    Returns:
        Response: A redirect response to the forgot password page, or the rendered forgot password page.
    """
    if request.method == 'POST':
        email: str = request.form.get('email')
        user: Optional[User] = User.query.filter_by(email=email).first()
        if user:
            token: str = serializer.dumps(email, salt='password-reset-salt')
            reset_link: str = url_for('reset_password', token=token, _external=True)
            reset_record: PasswordReset = PasswordReset(email=email, token=token)
            db.session.add(reset_record)
            db.session.commit()
            msg: Message = Message('Restablecer tu contraseña - Voy Consciente', recipients=[email])
            msg.body: str = f'Haz clic en el siguiente enlace para restablecer tu contraseña: {reset_link}\nEste enlace expira en 1 hora.'
            msg.html: str = f"""
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
def reset_password(token: str) -> Response:
    """
    Resets a user's password based on the provided token.

    Args:
        token (str): The password reset token for the user.

    Returns:
        Response: A redirect response to the login page, or the rendered reset password page.
    """
    try:
        email: str = serializer.loads(token, salt='password-reset-salt', max_age=3600)
        user: Optional[User] = User.query.filter_by(email=email).first()
        if not user:
            flash('Token inválido o usuario no encontrado.', 'error')
            return redirect(url_for('login'))
        if request.method == 'POST':
            new_password: str = request.form.get('password')
            confirm_password: str = request.form.get('confirm_password')
            if new_password != confirm_password:
                flash('Las contraseñas no coinciden. Por favor, intenta de nuevo.', 'error')
                return render_template('reset_password.html', token=token)
            password_pattern: re.Pattern = re.compile(r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$')
            if not password_pattern.match(new_password):
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
    reflexion = current_user.favorite_reflexiones.filter_by(id=reflexion_id).first()
    if reflexion:
        current_user.favorite_reflexiones.remove(reflexion)
        flash('Reflexión eliminada de favoritos.', 'success')
    else:
        current_user.favorite_reflexiones.append(Reflexion.query.get_or_404(reflexion_id))
        flash('Reflexión añadida a favoritos.', 'success')
    db.session.commit()
    return redirect(request.referrer or url_for('mostrar_reflexion', id=reflexion_id))

@app.route('/favoritos')
@login_required
def favoritos():
    return render_template('favoritos.html', favoritos=current_user.favorite_reflexiones.paginate(per_page=20))

@app.route('/download_pdf/<int:reflexion_id>')
def download_pdf(reflexion_id):
    reflexion = Reflexion.query.get_or_404(reflexion_id)
    html = render_template('reflexion_pdf.html', reflexion=reflexion, año_actual=datetime.now(tz=pytz.timezone('America/Argentina/Buenos_Aires')).year)
    pdf = HTML(string=html).write_pdf()
    return send_file(
        io.BytesIO(pdf),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'{reflexion.titulo}.pdf'
    )

from flask import render_template, jsonify, request, Response
from typing import Union, List, Dict

from flask import render_template, jsonify, request, Response
from typing import Union, List, Dict

@app.route('/consciencia', methods=['GET', 'POST'])
@login_required
def mostrar_consciencia() -> Union[str, Response]:
    """
    Display consciousness interface. Initializes session variables and processes 
    user interactions with the system.
    
    Returns:
        str: Rendered HTML template for GET requests.
        Response: JSON response for POST requests containing the interaction result.
    """
    # Initialize minimal session variables
    session_vars: List[str] = ['session_count', 'message_count', 'professional_mode', 'conversation_depth', 'emotional_state']

    for var in session_vars:
        if var not in session:
            session[var] = 0 if var.endswith('_count') else ('neutral' if var == 'emotional_state' else 0)

    # Increment session counter only on GET
    if request.method == 'GET':
        session['session_count'] += 1
        session['message_count'] = 0
        session['conversation_depth'] = 0

    identifier: int = current_user.id
    is_premium: bool = current_user.is_premium
    is_admin: bool = current_user.is_admin

    # Initialize or load user memory manager
    try:
        user_memory: UserMemoryManager = UserMemoryManager(current_user.id)
    except Exception as e:
        print(f"Error initializing UserMemoryManager: {e}")
        user_memory = UserMemoryManager(current_user.id)

    # Initialize personality engine
    personality_engine: PersonalityEngine = PersonalityEngine(current_user.id)

    # Initialize intention predictor
    intention_predictor: IntentionPredictor = IntentionPredictor()

    # Initialize meta-cognition system with custom thresholds
    meta_cognition: MetaCognitionSystem = MetaCognitionSystem(
        current_user.id,
        custom_thresholds={
            'relevance_threshold': 0.75,
            'coherence_threshold': 0.85,
            'elaboration_needed': 0.65,
            'correction_needed': 0.45
        }
    )

    # Interaction limits
    FREE_INTERACTION_LIMIT: int = 5
    interaction_count: int = Interaction.get_interaction_count(identifier, is_authenticated=True)
    remaining_interactions: int = 999999 if (is_premium or is_admin) else max(0, FREE_INTERACTION_LIMIT - interaction_count)

    # Class DespedidaDetector
    class DespedidaDetector:
        """Advanced system for detecting farewells and generating appropriate responses."""

        def __init__(self, umbral_confianza: float = 0.65):
            self.umbral_confianza: float = umbral_confianza
            self.patrones_despedida: Dict[str, List[str]] = {
                "es": [
                    r"\badi[óo]s\b", r"\bchao\b", r"\bhasta luego\b", r"\bhasta pronto\b",
                    r"\bnos vemos\b", r"\bhasta ma[nñ]ana\b", r"\bme voy\b", r"\bmuchas gracias\b",
                    r"\bgracias por todo\b", r"\bme despido\b", r"\bbuenas noches\b",
                    r"\bhasta la pr[óo]xima\b", r"\bfin\b", r"\bterminar\b", r"\bterminamos\b",
                    r"\beso es todo\b", r"\bya est[áa]\b", r"\blisto\b", r"\bes todo\b"
                ],
                "en": [
                    r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b", r"\blater\b", r"\bcya\b",
                    r"\bfarewell\b", r"\bhave a good\b", r"\btake care\b", r"\bending\b",
                    r"\bthank you for\b", r"\bthat's all\b", r"\bi'm done\b", r"\bthat will be all\b",
                    r"\bgood night\b", r"\bsigning off\b"
                ]
            }
            self.respuestas_despedida: Dict[str, List[str]] = {
                "es": [
                    "¡Hasta pronto! Ha sido un placer charlar contigo.",
                    "Gracias por conversar. ¡Que tengas un lindo día!",
                    "Entendido, me despido aquí. ¡Siempre un gusto ayudarte!",
                    "¡Perfecto! Nos vemos cuando quieras. ¡Adiós!",
                    "¡Hasta la próxima! Fue genial hablar contigo."
                ],
                "en": [
                    "See you later! It’s been great chatting with you.",
                    "Thanks for the talk. Have a wonderful day!",
                    "Got it, I’ll say goodbye here. Always a pleasure to assist!",
                    "Perfect! Catch you next time. Bye!",
                    "Until next time! Loved our conversation."
                ]
            }

        def detectar_idioma(self, texto: str) -> str:
            texto_lower = texto.lower()
            palabras_es = {"gracias", "hola", "adiós", "por", "favor", "buenos", "días"}
            palabras_en = {"thanks", "hello", "goodbye", "please", "good", "morning", "bye"}
            count_es = sum(texto_lower.count(palabra) for palabra in palabras_es)
            count_en = sum(texto_lower.count(palabra) for palabra in palabras_en)
            return "es" if count_es >= count_en else "en"

        def calcular_features(self, texto: str, historial: List[str], user_hour: float) -> Dict[str, float]:
            texto_lower = texto.lower()
            features = {}
            idioma = self.detectar_idioma(texto)
            patrones = self.patrones_despedida.get(idioma, self.patrones_despedida["en"])
            features["patron_explicito"] = min(sum(1 for patron in patrones if re.search(patron, texto_lower)) / 2, 1.0)
            features["brevedad"] = 1.0 if len(texto_lower.split()) <= 5 else (10 / len(texto_lower.split()) if len(texto_lower.split()) < 10 else 0.1)
            features["agradecimiento"] = any(a in texto_lower for a in {"gracias", "thank", "thanks", "agradec"})
            features["finalidad"] = any(f in texto_lower for f in {"eso es todo", "that's all", "por ahora", "listo", "done"})
            features["hora_nocturna"] = 0.3 if 19 <= user_hour or user_hour < 6 else 0.0
            features["longitud_conversacion"] = min(len(historial) / 20, 0.5) if historial else 0.0
            return features

    # Manejo de solicitudes
    if request.method == 'GET':
        # Renderizar la interfaz para GET
        return render_template('consciencia.html', 
                              remaining_interactions=remaining_interactions,
                              is_premium=is_premium,
                              is_admin=is_admin)
    elif request.method == 'POST':
        # Procesar datos enviados en POST
        data = request.get_json() or {}
        message = data.get('message', '')  # Mantenemos 'message' para coincidir con el frontend
    
        if not message:
            return jsonify({'error': 'No se proporcionó mensaje'}), 400  # Mensaje de error en español

        # Incrementar el contador de interacciones
        Interaction.increment(identifier, is_authenticated=True)

        # Detectar despedida
        despedida_detector = DespedidaDetector()
        historial = []  # Placeholder; reemplaza con el historial real si lo tienes
        user_hour = datetime.now(tz=timezone(timedelta(hours=-3))).hour
        features = despedida_detector.calcular_features(message, historial, user_hour)

        # Respuesta en español
        response = "¡Hola! ¿En qué puedo ayudarte hoy?" if message else "Acción registrada."
        return jsonify({
            'response': response,  # Mantenemos 'response' como clave para el frontend
            'features': features,
            'remaining_interactions': remaining_interactions - 1
    })
    else:
        # Caso por defecto para métodos no esperados
        return jsonify({'error': 'Method not allowed'}), 405

@app.route('/subscribe_info')
def subscribe_info():
    return render_template('subscribe_info.html', title='Suscripción - Voy Consciente')

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
        flash(f"Error al procesar la suscripción: {str(e)}", "error")
        return redirect(url_for('mostrar_consciencia'))

@app.route('/subscription_success')
@login_required
def subscription_success():
    current_user.is_premium = True
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
    reflexiones = Reflexion.query.order_by(Reflexion.id.desc()).limit(50).all()
    return render_template('editor.html', reflexiones=reflexiones)

@app.route('/editor/<int:id>', methods=['GET', 'POST'])
@login_required
def editar_reflexion(id):
    reflexion = Reflexion.query.get_or_404(id)
    if request.method == 'POST':
        updates = {key: request.form[key] for key in ['titulo', 'contenido', 'categoria', 'imagen', 'fecha'] if key in request.form}
        for key, value in updates.items():
            setattr(reflexion, key, value)
        try:
            db.session.commit()
            flash('Reflexión actualizada correctamente.', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error al actualizar: {str(e)}', 'error')
    return render_template('editar_reflexion.html', reflexion=reflexion)

def schedule_reflections():
    with app.app_context():
        users_by_frequency = {frequency: User.query.filter_by(frequency=frequency).all() for frequency in ['daily', 'weekly', 'monthly']}
        for frequency, users in users_by_frequency.items():
            if users:
                interval = {'daily': 1, 'weekly': 7, 'monthly': 30}[frequency]
                scheduler.add_job(func=send_weekly_reflection, trigger='interval', days=interval, id=f'reflection_{frequency}')

@app.route('/admin/users', methods=['GET', 'POST'])
@login_required
def admin_users() -> 'Response':
    """Handle admin user management, allowing retrieval and deletion of users.

    Returns:
        Response: A redirect or rendered template based on user actions.
    """
    if not current_user.is_admin:
        flash('No tienes permiso para acceder a esta página.', 'error')
        return redirect(url_for('home'))

    if request.method == 'POST':
        user_id: str = request.form.get('user_id', type=str)
        if not user_id:
            flash('No se proporcionó un ID de usuario para eliminar.', 'error')
            return redirect(url_for('admin_users'))

        user_to_delete: Optional[User] = User.query.get(user_id)
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

    users: List[User] = User.query.all()
    return render_template('admin_users.html', users=users)


def migrate_subscriber_to_user() -> None:
    """
    Migrates subscribers to users.

    :return: None
    """
    with app.app_context():
        if 'subscriber' in inspect(db.engine).get_table_names():
            old_subscribers: List[Tuple[str, datetime, Optional[str], Optional[str], Optional[bool], Optional[str]]] = db.session.execute(
                text('SELECT email, subscription_date, preferred_categories, frequency, push_enabled, onesignal_player_id FROM subscriber')
            ).fetchall()
            for sub in old_subscribers:
                user: Optional[User] = User.query.filter_by(email=sub[0]).first()
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
        

    scheduler = BackgroundScheduler()
    schedule_reflections()
    scheduler.start()
    app.run(
        host='0.0.0.0',
        port=5001,
        ssl_context=('cert.pem', 'key.pem'),
        debug=True
    )