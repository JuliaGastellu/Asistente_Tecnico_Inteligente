#!/usr/bin/env python3
"""
test_production.py — Suite de pruebas automatizadas contra el endpoint de producción.

Uso:
    python test_production.py

Requiere que BASE_URL apunte a la URL real de Render antes de ejecutar.
"""
import requests
import sys
import json
from datetime import datetime

# ⚠️ CAMBIAR ANTES DE EJECUTAR:
BASE_URL = "https://tu-app.onrender.com"   # <- reemplazar con URL real de Render

TIMEOUT = 30   # segundos — Render puede tener cold start

# Estados válidos de /health en esta app: "healthy" (todo OK) o "degraded"
# (la base vectorial no responde pero la API sigue arriba). No debe devolver 5xx.
VALID_HEALTH_STATUS = ("ok", "healthy", "degraded")


def test_health():
    """GET /health debe retornar status healthy o degraded (no 5xx)."""
    print("\n[1/5] GET /health ...")
    r = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "status" in data, f"Missing 'status' in response: {data}"
    assert data["status"] in VALID_HEALTH_STATUS, f"Unexpected status: {data['status']}"
    print(f"    ✅ status={data['status']}")
    return data


def test_query_rag():
    """POST /query con pregunta técnica debe retornar answer y sources."""
    print("\n[2/5] POST /query — consulta RAG ...")
    payload = {"query": "¿Cómo crear un middleware en FastAPI?"}
    r = requests.post(f"{BASE_URL}/query", json=payload, timeout=TIMEOUT)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "answer" in data, f"Missing 'answer': {data}"
    assert len(data["answer"]) > 20, f"Answer too short: {data['answer']}"
    print(f"    ✅ answer ({len(data['answer'])} chars), sources={data.get('sources', [])}")
    return data


def test_query_multitool():
    """POST /query con consulta multi-herramienta (RAG + calculadora)."""
    print("\n[3/5] POST /query — multi-tool ...")
    payload = {"query": "¿Qué es LangGraph y cuánto es 15% de 250?"}
    r = requests.post(f"{BASE_URL}/query", json=payload, timeout=TIMEOUT)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "answer" in data
    # Verificar que la calculadora fue invocada (37.5 debería aparecer)
    has_calc = "37.5" in data["answer"] or "37,5" in data["answer"]
    print(f"    ✅ respuesta recibida | calculadora invocada: {has_calc}")
    return data


def test_query_session():
    """POST /query con session_id para conversación continua."""
    print("\n[4/5] POST /query — session continua ...")
    session_id = f"test_session_{datetime.now().strftime('%H%M%S')}"
    payload = {"query": "¿Cómo añadir autenticación en FastAPI?", "session_id": session_id}
    r = requests.post(f"{BASE_URL}/query", json=payload, timeout=TIMEOUT)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "answer" in data
    print(f"    ✅ session_id={session_id} | respuesta recibida")
    return data


def test_invalid_request():
    """POST /query sin 'query' debe retornar 422 (validación Pydantic)."""
    print("\n[5/5] POST /query — validación de input ...")
    r = requests.post(f"{BASE_URL}/query", json={}, timeout=TIMEOUT)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"
    print(f"    ✅ 422 Unprocessable Entity (validación Pydantic OK)")


def main():
    print(f"{'='*60}")
    print(f"  Technical Documentation Assistant — Production Tests")
    print(f"  Target: {BASE_URL}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    tests = [
        test_health,
        test_query_rag,
        test_query_multitool,
        test_query_session,
        test_invalid_request,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"    ❌ FAILED: {e}")
            errors.append((test_fn.__name__, str(e)))
            failed += 1
        except requests.exceptions.ConnectionError:
            msg = f"No se pudo conectar a {BASE_URL} — ¿está el servidor corriendo?"
            print(f"    ❌ CONNECTION ERROR: {msg}")
            errors.append((test_fn.__name__, msg))
            failed += 1
        except requests.exceptions.Timeout:
            msg = f"Timeout después de {TIMEOUT}s — cold start de Render puede tardar más"
            print(f"    ❌ TIMEOUT: {msg}")
            errors.append((test_fn.__name__, msg))
            failed += 1

    print(f"\n{'='*60}")
    print(f"  RESULTADOS: {passed}/{len(tests)} tests pasaron")
    if errors:
        print(f"\n  Fallos:")
        for name, msg in errors:
            print(f"    • {name}: {msg}")

    if failed == 0:
        print(f"\n  🎉 ALL TESTS PASSED")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        print(f"\n  ⚠️  {failed} test(s) fallaron")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
