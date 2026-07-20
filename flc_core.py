"""
Feedback Loop Core (FLC)
Версия: 0.1.0
Дата: 2026-07-20
Автор: Эльшан Алиев
"""

import time
import json
from collections import deque
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class FLCSignal:
    """Формат сигнала обратной связи"""
    agent_id: str
    timestamp: str
    action_id: str
    anomaly_type: str  # "timeout" | "error_rate" | "state_delta"
    severity: float  # 0.0 - 1.0
    metrics: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class SensorLayer:
    """Сенсорный слой: сбор метрик после каждого действия"""
    
    def __init__(self, window_size: int = 10):
        self.metrics_history: Dict[str, deque] = {
            "response_time": deque(maxlen=window_size),
            "error_codes": deque(maxlen=window_size),
            "state_changes": deque(maxlen=window_size)
        }
        self.last_action_id: Optional[str] = None
    
    def collect(self, action_id: str, response_time_ms: float, 
                error_code: int = 0, state_change: float = 0.0) -> Dict[str, float]:
        """Собирает метрики после выполнения действия"""
        self.last_action_id = action_id
        self.metrics_history["response_time"].append(response_time_ms)
        self.metrics_history["error_codes"].append(error_code)
        self.metrics_history["state_changes"].append(state_change)
        
        return {
            "response_time_ms": response_time_ms,
            "error_code": error_code,
            "state_change": state_change
        }
    
    def get_recent_metrics(self) -> Dict[str, List[float]]:
        """Возвращает последние собранные метрики"""
        return {
            "response_time": list(self.metrics_history["response_time"]),
            "error_codes": list(self.metrics_history["error_codes"]),
            "state_changes": list(self.metrics_history["state_changes"])
        }


class AnalyticalLayer:
    """Аналитический слой: обнаружение аномалий через скользящее окно"""
    
    def __init__(self, window_size: int = 10, anomaly_threshold: float = 2.0):
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
    
    def detect_anomaly(self, metrics: Dict[str, float], 
                      history: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Анализирует метрики и возвращает оценку аномальности для каждого типа
        """
        anomalies = {}
        
        for metric_name, value in metrics.items():
            if metric_name not in history or not history[metric_name]:
                anomalies[metric_name] = 0.0
                continue
            
            # Вычисляем среднее и стандартное отклонение
            hist_values = history[metric_name]
            if len(hist_values) < 2:
                anomalies[metric_name] = 0.0
                continue
                
            mean = np.mean(hist_values)
            std = np.std(hist_values)
            
            if std == 0:
                anomalies[metric_name] = 0.0
            else:
                z_score = abs((value - mean) / std)
                # Нормализуем z-score в шкалу 0-1
                severity = min(z_score / self.anomaly_threshold, 1.0)
                anomalies[metric_name] = severity
        
        return anomalies
    
    def get_overall_severity(self, anomalies: Dict[str, float]) -> float:
        """Вычисляет общий уровень аномальности"""
        if not anomalies:
            return 0.0
        # Усредняем с весами (ошибки важнее времени)
        weights = {
            "response_time_ms": 0.3,
            "error_code": 0.6,
            "state_change": 0.1
        }
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if key in anomalies:
                weighted_sum += anomalies[key] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight


class ControlLayer:
    """Управляющий слой: принятие решения на основе аномалий"""
    
    def __init__(self, severity_threshold: float = 0.6):
        self.severity_threshold = severity_threshold
        self.is_paused = False
        self.last_signal: Optional[FLCSignal] = None
    
    def decide(self, agent_id: str, action_id: str, 
               severity: float, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Принимает решение: остановить, скорректировать или игнорировать
        """
        if severity >= self.severity_threshold:
            # Останавливаем цепочку
            self.is_paused = True
            signal = FLCSignal(
                agent_id=agent_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                action_id=action_id,
                anomaly_type=self._determine_anomaly_type(metrics),
                severity=severity,
                metrics=metrics
            )
            self.last_signal = signal
            
            return {
                "decision": "pause",
                "signal": signal.to_json(),
                "reason": f"Severity {severity:.2f} exceeds threshold {self.severity_threshold}"
            }
        else:
            return {
                "decision": "continue",
                "signal": None,
                "reason": f"Severity {severity:.2f} below threshold {self.severity_threshold}"
            }
    
    def _determine_anomaly_type(self, metrics: Dict[str, Any]) -> str:
        """Определяет тип аномалии на основе метрик"""
        if metrics.get("error_code", 0) >= 400:
            return "error_rate"
        if metrics.get("response_time_ms", 0) > 1000:
            return "timeout"
        if abs(metrics.get("state_change", 0.0)) > 10.0:
            return "state_delta"
        return "unknown"
    
    def reset(self):
        """Сбрасывает состояние (для нового цикла)"""
        self.is_paused = False
        self.last_signal = None


class FeedbackLoopCore:
    """Основной класс FLC - объединяет все слои"""
    
    def __init__(self, agent_id: str, window_size: int = 10, 
                 anomaly_threshold: float = 2.0, severity_threshold: float = 0.6):
        self.agent_id = agent_id
        self.sensor = SensorLayer(window_size=window_size)
        self.analyzer = AnalyticalLayer(
            window_size=window_size, 
            anomaly_threshold=anomaly_threshold
        )
        self.controller = ControlLayer(severity_threshold=severity_threshold)
        self.action_counter = 0
    
    def execute_action(self, action_func, *args, **kwargs) -> Dict[str, Any]:
        """
        Выполняет действие с встроенным контролем FLC
        """
        self.action_counter += 1
        action_id = f"action-{self.action_counter:03d}"
        
        # Выполняем действие и собираем метрики
        start_time = time.time()
        try:
            result = action_func(*args, **kwargs)
            error_code = 0
        except Exception as e:
            result = {"error": str(e)}
            error_code = 500
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Собираем метрики
        metrics = self.sensor.collect(
            action_id=action_id,
            response_time_ms=elapsed_ms,
            error_code=error_code,
            state_change=0.0  # В реальной системе здесь были бы данные о состоянии
        )
        
        # Анализируем аномалии
        history = self.sensor.get_recent_metrics()
        anomalies = self.analyzer.detect_anomaly(metrics, history)
        severity = self.analyzer.get_overall_severity(anomalies)
        
        # Принимаем решение
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
        """Сбрасывает состояние FLC"""
        self.controller.reset()
        self.action_counter = 0


# --- Пример использования ---
if __name__ == "__main__":
    print("🔄 Feedback Loop Core (FLC) Demo\n")
    
    # Создаём экземпляр FLC
    flc = FeedbackLoopCore(
        agent_id="demo-agent-001",
        window_size=5,
        anomaly_threshold=1.5,
        severity_threshold=0.5
    )
    
    # Симулируем действия агента
    def simulated_action(duration_ms: float = 200, should_fail: bool = False):
        """Симулирует действие агента (например, запрос к API)"""
        time.sleep(duration_ms / 1000.0)
        if should_fail:
            raise Exception("API connection timeout")
        return {"status": "success", "data": {"value": 42}}
    
    print("Выполняем последовательность действий...\n")
    
    # Серия действий: сначала нормальные, потом аномальные
    actions = [
        {"duration": 100, "fail": False},
        {"duration": 150, "fail": False},
        {"duration": 120, "fail": False},
        {"duration": 2000, "fail": False},  # Очень медленный ответ
        {"duration": 100, "fail": False},
        {"duration": 50, "fail": False},
        {"duration": 100, "fail": True},   # Ошибка
        {"duration": 100, "fail": False},
    ]
    
    for i, action_params in enumerate(actions, 1):
        duration = action_params["duration"]
        should_fail = action_params["fail"]
        
        print(f"Шаг {i}: запрос с таймаутом {duration}мс...")
        
        try:
            result = flc.execute_action(
                simulated_action,
                duration_ms=duration,
                should_fail=should_fail
            )
        except Exception as e:
            print(f"  ❌ Необработанная ошибка: {e}")
            continue
        
        # Выводим результат
        print(f"  Решение: {result['decision']}")
        print(f"  Severity: {result['severity']:.2f}")
        
        if result['decision'] == 'pause':
            print(f"  ⛔ ОСТАНОВКА ЦЕПОЧКИ!")
            print(f"  Причина: {result['reason']}")
            if result['signal']:
                print(f"  Сигнал FLC:\n{result['signal']}")
            break
        
        print(f"  ✅ Продолжаем...")
        print()
    
    print("\n--- Статистика ---")
    print(f"Всего выполнено действий: {flc.action_counter}")
    print(f"Состояние FLC: {'ОСТАНОВЛЕН' if flc.controller.is_paused else 'Активен'}")
    if flc.controller.is_paused:
        print(f"Причина остановки: {flc.controller.last_signal.anomaly_type}")
