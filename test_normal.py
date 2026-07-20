from flc_core import FeedbackLoopCore
import time

flc = FeedbackLoopCore(
    agent_id="test-agent",
    window_size=5,
    severity_threshold=0.3
)

def normal_action(delay_ms=100):
    """Нормальное действие без ошибок"""
    time.sleep(delay_ms / 1000.0)
    return {"status": "ok", "data": "test"}

def error_action(delay_ms=100):
    """Действие с ошибкой"""
    time.sleep(delay_ms / 1000.0)
    raise Exception("API error")

print("🔄 Тест 1: Нормальные действия (без ошибок)")
print("-" * 40)

for i in range(10):
    print(f"Шаг {i+1}: нормальное действие", end=" ")
    result = flc.execute_action(normal_action, 50)
    print(f"→ {result['decision']} (severity: {result['severity']:.2f})")
    
    if result['decision'] == 'pause':
        print("  ⛔ ОСТАНОВКА (не должно быть при нормальных действиях)")
        break

print(f"\nВыполнено: {flc.action_counter}")
print(f"Состояние: {'ОСТАНОВЛЕН' if flc.controller.is_paused else 'Активен'}")

print("\n" + "=" * 40)
print("🔄 Тест 2: Действие с ошибкой")
print("-" * 40)

flc.reset()

try:
    result = flc.execute_action(error_action, 100)
    print(f"Шаг 1: действие с ошибкой → {result['decision']} (severity: {result['severity']:.2f})")
    if result['decision'] == 'pause':
        print("  ⛔ ОСТАНОВКА (корректное поведение)")
except Exception as e:
    print(f"Шаг 1: ошибка → {e}")

print(f"\nВыполнено: {flc.action_counter}")
print(f"Состояние: {'ОСТАНОВЛЕН' if flc.controller.is_paused else 'Активен'}")
