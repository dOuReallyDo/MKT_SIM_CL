"""
Security Utilities Module.

Questo modulo contiene le classi e funzioni per la gestione della sicurezza.
"""

import os
import json
import base64
import logging
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from datetime import datetime, timedelta

# Configurazione del logger
logger = logging.getLogger('Security')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class CredentialManager:
    """Gestisce le credenziali in modo sicuro."""
    
    def __init__(self, encryption_key=None):
        """
        Inizializza il gestore delle credenziali.
        
        Args:
            encryption_key: Chiave di crittografia (opzionale)
        """
        self.encryption_key = encryption_key or self._generate_key()
        self.cipher_suite = Fernet(self.encryption_key.encode() if isinstance(self.encryption_key, str) else self.encryption_key)
    
    @staticmethod
    def _generate_key():
        """
        Genera una chiave di crittografia.
        
        Returns:
            str: Chiave di crittografia
        """
        key = Fernet.generate_key()
        return key.decode()
    
    def encrypt_credential(self, credential):
        """
        Cripta una credenziale.
        
        Args:
            credential: Credenziale da criptare
            
        Returns:
            str: Credenziale criptata
        """
        try:
            return self.cipher_suite.encrypt(credential.encode()).decode()
        except Exception as e:
            logger.error(f"Errore nella criptazione della credenziale: {e}")
            raise ValueError(f"Errore nella criptazione della credenziale: {e}")
    
    def decrypt_credential(self, encrypted_credential):
        """
        Decripta una credenziale.
        
        Args:
            encrypted_credential: Credenziale criptata
            
        Returns:
            str: Credenziale decriptata
        """
        try:
            return self.cipher_suite.decrypt(encrypted_credential.encode()).decode()
        except Exception as e:
            logger.error(f"Errore nella decrittazione della credenziale: {e}")
            raise ValueError(f"Errore nella decrittazione della credenziale: {e}")
    
    def save_credentials(self, credentials, file_path='credentials.enc'):
        """
        Salva le credenziali criptate su file.
        
        Args:
            credentials: Dizionario di credenziali
            file_path: Percorso del file
            
        Returns:
            bool: True se il salvataggio è riuscito, False altrimenti
        """
        try:
            encrypted_credentials = {
                key: self.encrypt_credential(value) if isinstance(value, str) else value
                for key, value in credentials.items()
            }
            with open(file_path, 'w') as f:
                json.dump(encrypted_credentials, f)
            return True
        except Exception as e:
            logger.error(f"Errore nel salvataggio delle credenziali: {e}")
            return False
    
    def load_credentials(self, file_path='credentials.enc'):
        """
        Carica le credenziali criptate da file.
        
        Args:
            file_path: Percorso del file
            
        Returns:
            dict: Dizionario di credenziali decriptate
        """
        try:
            with open(file_path, 'r') as f:
                encrypted_credentials = json.load(f)
            return {
                key: self.decrypt_credential(value) if isinstance(value, str) else value
                for key, value in encrypted_credentials.items()
            }
        except Exception as e:
            logger.error(f"Errore nel caricamento delle credenziali: {e}")
            return {}

class PasswordManager:
    """Gestisce le password in modo sicuro."""
    
    @staticmethod
    def hash_password(password, salt=None):
        """
        Genera l'hash di una password.
        
        Args:
            password: Password da hashare
            salt: Salt da utilizzare (opzionale)
            
        Returns:
            tuple: (hash, salt)
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key.decode(), base64.b64encode(salt).decode()
    
    @staticmethod
    def verify_password(password, password_hash, salt):
        """
        Verifica una password.
        
        Args:
            password: Password da verificare
            password_hash: Hash della password
            salt: Salt utilizzato per l'hash
            
        Returns:
            bool: True se la password è corretta, False altrimenti
        """
        salt_bytes = base64.b64decode(salt)
        hashed, _ = PasswordManager.hash_password(password, salt_bytes)
        return hashed == password_hash

class TokenManager:
    """Gestisce i token di autenticazione."""
    
    def __init__(self, secret_key=None):
        """
        Inizializza il gestore dei token.
        
        Args:
            secret_key: Chiave segreta per la firma dei token (opzionale)
        """
        self.secret_key = secret_key or os.urandom(32).hex()
    
    def generate_token(self, user_id, expiry_minutes=60):
        """
        Genera un token di autenticazione.
        
        Args:
            user_id: ID dell'utente
            expiry_minutes: Durata del token in minuti
            
        Returns:
            str: Token di autenticazione
        """
        expiry = datetime.now() + timedelta(minutes=expiry_minutes)
        expiry_str = expiry.strftime('%Y-%m-%d %H:%M:%S')
        
        payload = f"{user_id}:{expiry_str}"
        signature = hashlib.sha256(f"{payload}:{self.secret_key}".encode()).hexdigest()
        
        token = f"{base64.b64encode(payload.encode()).decode()}.{signature}"
        return token
    
    def validate_token(self, token):
        """
        Valida un token di autenticazione.
        
        Args:
            token: Token da validare
            
        Returns:
            dict: Informazioni contenute nel token o None se il token non è valido
        """
        try:
            # Estrai payload e firma
            payload_b64, signature = token.split('.')
            payload = base64.b64decode(payload_b64).decode()
            
            # Verifica firma
            expected_signature = hashlib.sha256(f"{payload}:{self.secret_key}".encode()).hexdigest()
            if signature != expected_signature:
                logger.warning("Firma del token non valida")
                return None
            
            # Estrai informazioni dal payload
            user_id, expiry_str = payload.split(':')
            expiry = datetime.strptime(expiry_str, '%Y-%m-%d %H:%M:%S')
            
            # Verifica scadenza
            if datetime.now() > expiry:
                logger.warning("Token scaduto")
                return None
            
            return {
                'user_id': user_id,
                'expiry': expiry
            }
        except Exception as e:
            logger.error(f"Errore nella validazione del token: {e}")
            return None

def generate_random_key(length=32):
    """
    Genera una chiave casuale.
    
    Args:
        length: Lunghezza della chiave in byte
        
    Returns:
        str: Chiave casuale in formato esadecimale
    """
    return os.urandom(length).hex()

def secure_hash(data, salt=None):
    """
    Genera un hash sicuro dei dati.
    
    Args:
        data: Dati da hashare
        salt: Salt da utilizzare (opzionale)
        
    Returns:
        tuple: (hash, salt)
    """
    if salt is None:
        salt = os.urandom(16)
    
    data_bytes = data.encode() if isinstance(data, str) else data
    
    h = hashlib.sha256()
    h.update(salt)
    h.update(data_bytes)
    
    return h.hexdigest(), base64.b64encode(salt).decode() 