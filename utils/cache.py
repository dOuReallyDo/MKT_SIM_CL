"""
Cache Utilities Module.

Questo modulo contiene le classi e funzioni per la gestione della cache.
"""

import os
import pickle
import logging
import time
import json
from functools import wraps
from pathlib import Path

# Importazione condizionale di Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Redis non è installato. La cache Redis non sarà disponibile.")

# Configurazione del logger
logger = logging.getLogger('Cache')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class MemoryCache:
    """Cache in memoria."""
    
    def __init__(self, max_size=1000):
        """
        Inizializza la cache in memoria.
        
        Args:
            max_size: Dimensione massima della cache
        """
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, key):
        """
        Recupera un valore dalla cache.
        
        Args:
            key: Chiave del valore da recuperare
            
        Returns:
            Il valore recuperato o None se non trovato
        """
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key, value, ttl=None):
        """
        Salva un valore nella cache.
        
        Args:
            key: Chiave del valore
            value: Valore da salvare
            ttl: Tempo di vita del valore in secondi (non usato)
            
        Returns:
            True se il salvataggio è riuscito, False altrimenti
        """
        # Controlla se la cache ha raggiunto la dimensione massima
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Rimuovi l'elemento meno recentemente usato
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
        return True
    
    def delete(self, key):
        """
        Elimina un valore dalla cache.
        
        Args:
            key: Chiave del valore da eliminare
            
        Returns:
            True se l'eliminazione è riuscita, False altrimenti
        """
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            return True
        return False
    
    def clear(self):
        """
        Pulisce la cache.
        
        Returns:
            True
        """
        self.cache.clear()
        self.access_times.clear()
        return True

class DiskCache:
    """Cache su disco."""
    
    def __init__(self, cache_dir='cache'):
        """
        Inizializza la cache su disco.
        
        Args:
            cache_dir: Directory per la cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key):
        """
        Ottiene il percorso del file di cache per una chiave.
        
        Args:
            key: Chiave del valore
            
        Returns:
            Percorso del file di cache
        """
        # Converti la chiave in un nome di file valido
        safe_key = str(key).replace('/', '_').replace('\\', '_')
        return self.cache_dir / f"{safe_key}.cache"
    
    def get(self, key):
        """
        Recupera un valore dalla cache.
        
        Args:
            key: Chiave del valore da recuperare
            
        Returns:
            Il valore recuperato o None se non trovato
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Controlla se il valore è scaduto
                if 'expire_time' in cache_data and cache_data['expire_time'] < time.time():
                    self.delete(key)
                    return None
                
                return cache_data['value']
            except Exception as e:
                logger.error(f"Errore nel recupero della cache: {e}")
                return None
        return None
    
    def set(self, key, value, ttl=None):
        """
        Salva un valore nella cache.
        
        Args:
            key: Chiave del valore
            value: Valore da salvare
            ttl: Tempo di vita del valore in secondi
            
        Returns:
            True se il salvataggio è riuscito, False altrimenti
        """
        cache_path = self._get_cache_path(key)
        try:
            cache_data = {
                'value': value,
                'created_time': time.time()
            }
            
            if ttl is not None:
                cache_data['expire_time'] = time.time() + ttl
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            return True
        except Exception as e:
            logger.error(f"Errore nel salvataggio della cache: {e}")
            return False
    
    def delete(self, key):
        """
        Elimina un valore dalla cache.
        
        Args:
            key: Chiave del valore da eliminare
            
        Returns:
            True se l'eliminazione è riuscita, False altrimenti
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                os.remove(cache_path)
                return True
            except Exception as e:
                logger.error(f"Errore nell'eliminazione della cache: {e}")
                return False
        return False
    
    def clear(self):
        """
        Pulisce la cache.
        
        Returns:
            True se la pulizia è riuscita, False altrimenti
        """
        try:
            for cache_file in self.cache_dir.glob('*.cache'):
                os.remove(cache_file)
            return True
        except Exception as e:
            logger.error(f"Errore nella pulizia della cache: {e}")
            return False

class RedisCache:
    """Cache distribuita basata su Redis."""
    
    def __init__(self, host='localhost', port=6379, db=0, **kwargs):
        """
        Inizializza la cache Redis.
        
        Args:
            host: Host del server Redis
            port: Porta del server Redis
            db: Indice del database Redis
            **kwargs: Parametri aggiuntivi per Redis
        """
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=False,
                **kwargs
            )
            # Verifica la connessione
            self.redis_client.ping()
        except Exception as e:
            self.logger.error(f"Errore nella connessione a Redis: {e}")
            self.redis_client = None
            self.logger.warning("Redis non è disponibile. La cache Redis non sarà utilizzata.")
    
    def get(self, key):
        """
        Recupera un valore dalla cache.
        
        Args:
            key: Chiave del valore da recuperare
            
        Returns:
            Il valore recuperato o None se non trovato
        """
        try:
            if not self.redis_client:
                return None
            
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            self.logger.error(f"Errore nel recupero dalla cache: {e}")
            return None
    
    def set(self, key, value, ttl=3600):
        """
        Salva un valore nella cache.
        
        Args:
            key: Chiave del valore
            value: Valore da salvare
            ttl: Tempo di vita del valore in secondi
            
        Returns:
            True se il salvataggio è riuscito, False altrimenti
        """
        try:
            if not self.redis_client:
                return False
            
            data = pickle.dumps(value)
            return self.redis_client.setex(key, ttl, data)
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio nella cache: {e}")
            return False
    
    def delete(self, key):
        """
        Elimina un valore dalla cache.
        
        Args:
            key: Chiave del valore da eliminare
            
        Returns:
            True se l'eliminazione è riuscita, False altrimenti
        """
        try:
            if not self.redis_client:
                return False
            
            return bool(self.redis_client.delete(key))
        except Exception as e:
            self.logger.error(f"Errore nell'eliminazione dalla cache: {e}")
            return False
    
    def clear(self):
        """
        Pulisce la cache.
        
        Returns:
            True se la pulizia è riuscita, False altrimenti
        """
        try:
            if not self.redis_client:
                return False
            
            self.redis_client.flushdb()
            return True
        except Exception as e:
            self.logger.error(f"Errore nella pulizia della cache: {e}")
            return False

class CacheManager:
    """Gestore di cache che utilizza diverse strategie di caching."""
    
    def __init__(self, use_memory=True, use_disk=True, use_redis=True, **kwargs):
        """
        Inizializza il gestore di cache.
        
        Args:
            use_memory: Utilizzare la cache in memoria
            use_disk: Utilizzare la cache su disco
            use_redis: Utilizzare la cache Redis
            **kwargs: Parametri aggiuntivi per le cache
        """
        self.caches = []
        
        if use_memory:
            memory_kwargs = kwargs.get('memory', {})
            self.caches.append(MemoryCache(**memory_kwargs))
        
        if use_disk:
            disk_kwargs = kwargs.get('disk', {})
            self.caches.append(DiskCache(**disk_kwargs))
        
        # Usa Redis solo se disponibile
        if use_redis and REDIS_AVAILABLE:
            redis_kwargs = kwargs.get('redis', {})
            self.caches.append(RedisCache(**redis_kwargs))
    
    def get(self, key):
        """
        Recupera un valore dalla cache.
        
        Args:
            key: Chiave del valore da recuperare
            
        Returns:
            Il valore recuperato o None se non trovato
        """
        for cache in self.caches:
            value = cache.get(key)
            if value is not None:
                # Propaga il valore alle cache precedenti
                for prev_cache in self.caches:
                    if prev_cache != cache:
                        prev_cache.set(key, value)
                return value
        return None
    
    def set(self, key, value, ttl=None):
        """
        Salva un valore in tutte le cache.
        
        Args:
            key: Chiave del valore
            value: Valore da salvare
            ttl: Tempo di vita del valore in secondi
            
        Returns:
            True se il salvataggio è riuscito in almeno una cache, False altrimenti
        """
        success = False
        for cache in self.caches:
            if cache.set(key, value, ttl):
                success = True
        return success
    
    def delete(self, key):
        """
        Elimina un valore da tutte le cache.
        
        Args:
            key: Chiave del valore da eliminare
            
        Returns:
            True se l'eliminazione è riuscita in almeno una cache, False altrimenti
        """
        success = False
        for cache in self.caches:
            if cache.delete(key):
                success = True
        return success
    
    def clear(self):
        """
        Pulisce tutte le cache.
        
        Returns:
            True se la pulizia è riuscita in almeno una cache, False altrimenti
        """
        success = False
        for cache in self.caches:
            if cache.clear():
                success = True
        return success

def memoize(ttl=None):
    """
    Decoratore per memorizzare i risultati di una funzione.
    
    Args:
        ttl: Tempo di vita dei risultati in secondi
    
    Returns:
        Decoratore
    """
    def decorator(func):
        cache = {}
        expire_times = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Crea una chiave di cache basata sugli argomenti
            key = pickle.dumps((args, sorted(kwargs.items())))
            
            # Controlla se il risultato è nella cache ed è ancora valido
            current_time = time.time()
            if key in cache and (ttl is None or expire_times[key] > current_time):
                return cache[key]
            
            # Esegui la funzione e memorizza il risultato
            result = func(*args, **kwargs)
            cache[key] = result
            
            if ttl is not None:
                expire_times[key] = current_time + ttl
            
            return result
        
        # Funzione per pulire la cache
        def clear_cache():
            cache.clear()
            expire_times.clear()
        
        # Aggiungi la funzione di pulizia come attributo del wrapper
        wrapper.clear_cache = clear_cache
        
        return wrapper
    
    return decorator 