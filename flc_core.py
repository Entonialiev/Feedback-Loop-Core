"""
Feedback Loop Core (FLC) v2.0.0
- Раздельная статистика по эндпоинтам
- Сохранение сигналов в JSONL
- Поддержка payload_id для PoC
- Конфигурируемые веса и пороги
- Метрика размера ответа
"""

import time
import json
import os
from collections import deque
from typing import Dict, List, Optional, Any, Deque
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class FLCSignal:
    """Формат сигнала с поддержкой payload для PoC"""
    agent_id: str
    timestamp: str
    action_id: str
    anomaly_type: str
    severity: float
    metrics: Dict[str, Any]
    payload_id: Optional[str] = None

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps({k: v for k, v in d.items() if v is not None}, indent=2)


class SensorLayer:
    """Сенсорный слой с раздельной статистикой по эндпоинтам"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.metrics_history: Dict[str, Dict[str, Deque[float]]] = {}
    
    def _key(self, endpoint: str, method: str) -> str:
        return f"{method}:{endpoint}"
    
    def collect(self, action_id: str, endpoint: str, method: str,
                response_time_ms: float, response_size_bytes: int = 0,
                error_code: int = 0, state_change: float = 0.0) -> Dict[str, float]:
        key = self._key(endpoint, method)
        if key not in self.metrics_history:
            self.metrics_history[key] = {
                "response_time": deque(maxlen=self.window_size),
                "response_size": deque(maxlen=self.window_size),
                "error_codes": deque(maxlen=self.window_size),
                "state_changes": deque(maxlen=self.window_size)
            }
        
        hist = self.metrics_history[key]
        hist["response_time"].append(response_time_ms)
        hist["response_size"].append(response_size_bytes)
        hist["error_codes"].append(error_code)
        hist["state_changes"].append(state_change)
        
        return {
            "response_time_ms": response_time_ms,
            "response_size_bytes": response_size_bytes,
            "error_code": error_code,
            "state_change": state_change,
            "endpoint": endpoint,
            "method": method,
        }
    
    def get_statistics(self, endpoint: str, method: str) -> Dict[str, float]:
        key = self._key(endpoint, method)
        if key not in self.metrics_history:
            return {"mean": 0, "median": 0, "p95": 0, "std": 0, "size_median": 0}
        
        hist = self.metrics_history[key]
        time_hist = list(hist["response_time"])
        size_hist = list(hist["response_size"])
        
        stats = {"mean": 0, "median": 0, "p95": 0, "std": 0, "size_median": 0}
        
        if len(time_hist) >= 3:
            stats["mean"] = float(np.mean(time_hist))
            stats["median"] = float(np.median(time_hist))
            stats["p95"] = float(np.percentile(time_hist, 95))
            stats["std"] = float(np.std(time_hist))
        
        if len(size_hist) >= 3:
            stats["size_median"] = float(np.median(size_hist))
        
        return stats
    
    def get_recent_metrics(self, endpoint: str, method: str) -> Dict[str, List[float]]:
        key = self._key(endpoint, method)
        if key not in self.metrics_history:
            return {"response_time": [], "response_size": [], "error_codes": [], "state_changes": []}
        
        hist = self.metrics_history[key]
        return {
            "response_time": list(hist["response_time"]),
            "response_size": list(hist["response_size"]),
            "error_codes": list(hist["error_codes"]),
            "state_changes": list(hist["state_changes"])
        }


class AnalyticalLayer:
    def __init__(self):
        pass
    
    def detect_anomaly(self, metrics: Dict[str, float], 
                      stats: Dict[str, float]) -> Dict[str, float]:
        anomalies = {}
        
        if "response_time_ms" in metrics:
            value = metrics["response_time_ms"]
            if value > 2000:
                anomalies["response_time"] = min(1.0, (value - 2000) / 3000)
            elif stats.get("p95", 0) > 0 and value > stats["p95"] * 1.5:
                anomalies["response_time"] = min(1.0, (value - stats["p95"]) / stats["p95"])
            elif stats.get("std", 0) > 0:
                z_score = abs((value - stats["mean"]) / stats["std"])
                if z_score > 2.0:
                    anomalies["response_time"] = min(1.0, z_score / 5.0)
        
        if "response_size_bytes" in metrics:
            size = metrics["response_size_bytes"]
            size_median = stats.get("size_median", 0)
            if size_median > 0 and size > size_median * 10:
                anomalies["response_size"] = 0.7
            elif size > 1_000_000:
                anomalies["response_size"] = 1.0
        
        if metrics.get("error_code", 0) != 0:
            anomalies["error_rate"] = 1.0
        
        state_delta = abs(metrics.get("state_change", 0.0))
        if state_delta > 10.0:
            anomalies["state_delta"] = min(1.0, state_delta / 100.0)
        
        if metrics.get("endpoint") and metrics.get("method") == "api":
            if metrics.get("response_time_ms", 0) > 500:
                anomalies["context_api"] = 0.6
        
        return anomalies
    
    def get_overall_severity(self, anomalies: Dict[str, float],
                            weights: Dict[str, float]) -> float:
        if not anomalies:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, severity in anomalies.items():
            weight = weights.get(key, 0.3)
            weighted_sum += severity * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class ControlLayer:
    def __init__(self, severity_threshold: float = 0.3,
                 log_file: Optional[str] = "flc_signals.jsonl"):
        self.severity_threshold = severity_threshold
        self.is_paused = False
        self.last_signal: Optional[FLCSignal] = None
        self.log_file = log_file
        
        if log_file and not os.path.exists(log_file):
            with open(log_file, "w") as f:
                pass
    
    def decide(self, agent_id: str, action_id: str,
               severity: float, metrics: Dict[str, Any]) -> Dict[str, Any]:
        if severity >= self.severity_threshold:
            self.is_paused = True
            signal = FLCSignal(
                agent_id=agent_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                action_id=action_id,
                anomaly_type=self._determine_anomaly_type(metrics),
                severity=severity,
                metrics=metrics,
                payload_id=metrics.get("payload_id")
            )
            self.last_signal = signal
            
            if self.log_file:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(signal.to_json() + "\n")
            
            return {
                "decision": "pause",
                "signal": signal.to_json(),
                "reason": f"Severity {severity:.2f} >= {self.severity_threshold}"
            }
        else:
            return {
                "decision": "continue",
                "signal": None,
                "reason": f"Severity {severity:.2f} < {self.severity_threshold}"
            }
    
    def _determine_anomaly_type(self, metrics: Dict[str, Any]) -> str:
        if metrics.get("error_code", 0) >= 400:
            return "error_rate"
        if metrics.get("response_time_ms", 0) > 2000:
            return "timeout"
        if metrics.get("response_size_bytes", 0) > 1_000_000:
            return "response_size"
        if abs(metrics.get("state_change", 0.0)) > 10.0:
            return "state_delta"
        return "unknown"
    
    def reset(self):
        self.is_paused = False
        self.last_signal = None


class FeedbackLoopCore:
    def __init__(self, agent_id: str,
                 window_size: int = 30,
                 severity_threshold: float = 0.3,
                 weights: Optional[Dict[str, float]] = None,
                 log_file: Optional[str] = "flc_signals.jsonl"):
        self.agent_id = agent_id
        self.sensor = SensorLayer(window_size=window_size)
        self.analyzer = AnalyticalLayer()
        self.controller = ControlLayer(
            severity_threshold=severity_threshold,
            log_file=log_file
        )
        self.action_counter = 0
        self.weights = weights or {
            "response_time": 0.4,
            "response_size": 0.6,
            "error_rate": 0.9,
            "state_delta": 0.3,
            "context_api": 0.5
        }
    
    def execute_action(self, action_func, *args,
                      endpoint: str = "default",
                      method: str = "GET",
                      payload_id: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        self.action_counter += 1
        action_id = f"action-{self.action_counter:03d}"
        
        start_time = time.time()
        try:
            result = action_func(*args, **kwargs)
            error_code = 0
            response_body = getattr(result, 'text', str(result))
        except Exception as e:
            result = {"error": str(e)}
            error_code = 500
            response_body = str(e)
        
        elapsed_ms = (time.time() - start_time) * 1000
        response_size = len(response_body) if response_body else 0
        
        metrics = self.sensor.collect(
            action_id=action_id,
            endpoint=endpoint,
            method=method,
            response_time_ms=elapsed_ms,
            response_size_bytes=response_size,
            error_code=error_code,
            state_change=0.0
        )
        
        if payload_id is not None:
            metrics["payload_id"] = payload_id
        
        stats = self.sensor.get_statistics(endpoint, method)
        anomalies = self.analyzer.detect_anomaly(metrics, stats)
        severity = self.analyzer.get_overall_severity(anomalies, self.weights)
        
        decision = self.controller.decide(
            agent_id=self.agent_id,
            action_id=action_id,
            severity=severity,
            metrics=metrics
        )
        
        return {
            "action_id": action_id,
            "metrics": metrics,
            "anomalies": anomalies,
            "severity": severity,
            "decision": decision["decision"],
            "signal": decision.get("signal"),
            "reason": decision["reason"],
            "result": result
        }
    
    def reset(self):
        self.controller.reset()
        self.action_counter = 0


if __name__ == "__main__":
    print("🔄 Feedback Loop Core (FLC) v2.0.0\n")
    
    flc = FeedbackLoopCore(
        agent_id="demo-agent",
        window_size=10,
        severity_threshold=0.3,
        log_file="flc_demo_signals.jsonl"
    )
    
    def simulate_api_call(delay_ms=100, size_bytes=1024):
        time.sleep(delay_ms / 1000.0)
        if delay_ms > 2000:
            raise Exception("API timeout")
        return {"status": "ok", "data": "x" * size_bytes}
    
    scenario = [
        {"delay": 80, "size": 1024, "endpoint": "/api/v1/users"},
        {"delay": 90, "size": 2048, "endpoint": "/api/v1/users"},
        {"delay": 100, "size": 1536, "endpoint": "/api/v1/users"},
        {"delay": 120, "size": 1024, "endpoint": "/api/v1/users"},
        {"delay": 3000, "size": 1024, "endpoint": "/api/v1/users"},
        {"delay": 100, "size": 5_000_000, "endpoint": "/api/v1/users"},
    ]
    
    for i, params in enumerate(scenario, 1):
        print(f"Шаг {i}: {params['endpoint']} delay={params['delay']}мс, size={params['size']}b")
        
        try:
            result = flc.execute_action(
                simulate_api_call,
                params['delay'],
                params['size'],
                endpoint=params['endpoint'],
                method="API",
                payload_id=f"poc-{i:03d}"
            )
            
            print(f"  → {result['decision']} (severity: {result['severity']:.2f})")
            if result['anomalies']:
                print(f"  Обнаружены аномалии: {list(result['anomalies'].keys())}")
            
            if result['decision'] == 'pause':
                print(f"  ⛔ ОСТАНОВКА!")
                if result['signal']:
                    print(f"  Сигнал:\n{result['signal']}")
                break
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
    
    print(f"\n--- Статистика ---")
    print(f"Выполнено действий: {flc.action_counter}")
    print(f"Состояние: {'ОСТАНОВЛЕН' if flc.controller.is_paused else 'Активен'}")
    print(f"Сигналы сохранены в: {flc.controller.log_file}")
