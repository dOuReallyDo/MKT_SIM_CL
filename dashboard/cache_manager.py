"""
Cache Manager Module.

Questo modulo fornisce un sistema di cache condivisa per l'accesso rapido ai dati
tra i vari componenti del sistema.
"""

import os
import json
import time
import logging
import threading
import hashlib
from typing import Dict, Any, Optional, Tuple, List

class CacheManager:
    """Gestisce la cache condivisa del sistema"""
    
    def __init__(self, cache_dir='./dashboard/cache'):
        """
        Inizializza il gestore della cache
        
        Args:
            cache_dir: Directory in cui salvare i file di cache
        """
        self.cache_dir = cache_dir
        self.lock = threading.RLock()
        self.memory_cache = {}
        self.expiry_times = {}
        
        # Configura il logging
        self.logger = logging.getLogger('CacheManager')
        self.logger.setLevel(logging.INFO)
        
        # Se non ci sono handler, aggiungine uno
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
        
        # Crea la directory cache se non esiste
        os.makedirs(cache_dir, exist_ok=True)
        
        # Avvia la pulizia periodica
        self._start_cleanup_thread()
        
        self.logger.info(f"Cache Manager inizializzato in {cache_dir}")
    
    def _start_cleanup_thread(self):
        """Avvia un thread per la pulizia periodica della cache"""
        def cleanup_task():
            while True:
                try:
                    self._cleanup_expired_items()
                    time.sleep(300)  # Pulisci ogni 5 minuti
                except Exception as e:
                    self.logger.error(f"Errore nel thread di pulizia: {e}")
                    time.sleep(60)  # In caso di errore, riprova dopo 1 minuto
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
        self.logger.info("Thread di pulizia della cache avviato")
    
    def _cleanup_expired_items(self):
        """Rimuove gli elementi scaduti dalla cache"""
        with self.lock:
            current_time = time.time()
            # Pulisci la cache in memoria
            expired_keys = [k for k, v in self.expiry_times.items() if v < current_time]
            for key in expired_keys:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                if key in self.expiry_times:
                    del self.expiry_times[key]
            
            # Pulisci la cache su disco
            if os.path.exists(self.cache_dir):
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(self.cache_dir, filename)
                        try:
                            # Controlla se il file è scaduto
                            if self._is_cache_file_expired(file_path):
                                os.remove(file_path)
                                self.logger.debug(f"Rimosso file di cache scaduto: {filename}")
                        except Exception as e:
                            self.logger.error(f"Errore nella pulizia del file {filename}: {e}")
    
    def _is_cache_file_expired(self, file_path):
        """
        Verifica se un file di cache è scaduto
        
        Args:
            file_path: Percorso del file di cache
            
        Returns:
            True se il file è scaduto, False altrimenti
        """
        try:
            with open(file_path, 'r') as f:
                cache_data = json.load(f)
                if 'expiry' in cache_data:
                    return time.time() > cache_data['expiry']
            return False
        except Exception:
            # Se ci sono errori nella lettura, considera il file come scaduto
            return True
    
    def _get_cache_file_path(self, key):
        """
        Ottiene il percorso del file di cache per una chiave
        
        Args:
            key: Chiave della cache
            
        Returns:
            Percorso del file di cache
        """
        # Crea un hash della chiave per usarlo come nome file
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.json")
    
    def get(self, key: str, default=None) -> Any:
        """
        Recupera un valore dalla cache
        
        Args:
            key: Chiave della cache
            default: Valore predefinito se la chiave non esiste
            
        Returns:
            Valore associato alla chiave o default se non trovato
        """
        # Prima controlla la cache in memoria
        with self.lock:
            if key in self.memory_cache:
                # Verifica se l'elemento è scaduto
                if key in self.expiry_times and self.expiry_times[key] < time.time():
                    del self.memory_cache[key]
                    del self.expiry_times[key]
                    self.logger.debug(f"Elemento scaduto rimosso dalla cache: {key}")
                else:
                    self.logger.debug(f"Cache hit in memoria per {key}")
                    return self.memory_cache[key]
        
        # Se non in memoria, controlla la cache su disco
        file_path = self._get_cache_file_path(key)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    cache_data = json.load(f)
                    
                    # Verifica se l'elemento è scaduto
                    if 'expiry' in cache_data and cache_data['expiry'] < time.time():
                        os.remove(file_path)
                        self.logger.debug(f"File di cache scaduto rimosso: {key}")
                        return default
                    
                    # Memorizza anche in cache di memoria per accessi futuri
                    if 'data' in cache_data:
                        with self.lock:
                            self.memory_cache[key] = cache_data['data']
                            if 'expiry' in cache_data:
                                self.expiry_times[key] = cache_data['expiry']
                        
                        self.logger.debug(f"Cache hit su disco per {key}")
                        return cache_data['data']
            except Exception as e:
                self.logger.error(f"Errore nella lettura della cache per {key}: {e}")
        
        self.logger.debug(f"Cache miss per {key}")
        return default
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Salva un valore nella cache
        
        Args:
            key: Chiave della cache
            value: Valore da salvare
            ttl: Tempo di vita in secondi (default: 1 ora)
            
        Returns:
            True se l'operazione ha avuto successo, False altrimenti
        """
        try:
            expiry_time = time.time() + ttl
            
            # Salva in memoria
            with self.lock:
                self.memory_cache[key] = value
                self.expiry_times[key] = expiry_time
            
            # Salva su disco
            file_path = self._get_cache_file_path(key)
            cache_data = {
                'data': value,
                'expiry': expiry_time,
                'created_at': time.time()
            }
            
            with open(file_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.logger.debug(f"Elemento salvato in cache: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio in cache per {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Elimina un elemento dalla cache
        
        Args:
            key: Chiave della cache
            
        Returns:
            True se l'elemento è stato eliminato, False altrimenti
        """
        try:
            # Elimina dalla memoria
            with self.lock:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                if key in self.expiry_times:
                    del self.expiry_times[key]
            
            # Elimina dal disco
            file_path = self._get_cache_file_path(key)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            self.logger.debug(f"Elemento eliminato dalla cache: {key}")
            return True
        except Exception as e:
            self.logger.error(f"Errore nell'eliminazione dalla cache per {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Svuota completamente la cache
        
        Returns:
            True se l'operazione ha avuto successo, False altrimenti
        """
        try:
            # Svuota la memoria
            with self.lock:
                self.memory_cache.clear()
                self.expiry_times.clear()
            
            # Svuota il disco
            if os.path.exists(self.cache_dir):
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.json'):
                        os.remove(os.path.join(self.cache_dir, filename))
            
            self.logger.info("Cache completamente svuotata")
            return True
        except Exception as e:
            self.logger.error(f"Errore nello svuotamento della cache: {e}")
            return False
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Recupera più valori dalla cache
        
        Args:
            keys: Lista di chiavi da recuperare
            
        Returns:
            Dizionario con le chiavi trovate e i relativi valori
        """
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    def set_many(self, items: Dict[str, Any], ttl: int = 3600) -> bool:
        """
        Salva più valori nella cache
        
        Args:
            items: Dizionario con chiavi e valori da salvare
            ttl: Tempo di vita in secondi (default: 1 ora)
            
        Returns:
            True se tutte le operazioni hanno avuto successo, False altrimenti
        """
        success = True
        for key, value in items.items():
            if not self.set(key, value, ttl):
                success = False
        return success
    
    def delete_many(self, keys: List[str]) -> bool:
        """
        Elimina più elementi dalla cache
        
        Args:
            keys: Lista di chiavi da eliminare
            
        Returns:
            True se tutte le operazioni hanno avuto successo, False altrimenti
        """
        success = True
        for key in keys:
            if not self.delete(key):
                success = False
        return success
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Ottiene statistiche sulla cache
        
        Returns:
            Dizionario con le statistiche della cache
        """
        memory_items_count = len(self.memory_cache)
        
        disk_items_count = 0
        disk_size_bytes = 0
        if os.path.exists(self.cache_dir):
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            disk_items_count = len(cache_files)
            
            for filename in cache_files:
                file_path = os.path.join(self.cache_dir, filename)
                disk_size_bytes += os.path.getsize(file_path)
        
        return {
            'memory_items_count': memory_items_count,
            'disk_items_count': disk_items_count,
            'disk_size_bytes': disk_size_bytes,
            'disk_size_mb': round(disk_size_bytes / (1024 * 1024), 2)
        } 