"""
═══════════════════════════════════════════════════════════════════
AGENTE DE SEÑALES FUTUROS — MNQ · MES · MYM
Estrategia: EMA9/EMA20/EMA200 + Pullback + Volumen + VWAP
Análisis: Claude AI (Anthropic)
Datos: yfinance (gratuito)
Alertas: Telegram Bot
Deploy: Railway (un solo archivo)
═══════════════════════════════════════════════════════════════════

VARIABLES DE ENTORNO (configurar en Railway):
  ANTHROPIC_API_KEY   = sk-ant-api03-...
  TELEGRAM_BOT_TOKEN  = 123456:ABC-...
  TELEGRAM_CHAT_ID    = -1001234567890  (grupo) o 123456789 (personal)

INSTALACIÓN EN RAILWAY:
  1. Sube este archivo como main.py
  2. Sube requirements.txt con el contenido indicado al final
  3. Agrega las 3 variables de entorno
  4. Railway despliega automáticamente
═══════════════════════════════════════════════════════════════════
"""

# ── requirements.txt (crear archivo separado en Railway) ──────────────────
# fastapi==0.115.5
# uvicorn==0.32.1
# anthropic==0.40.0
# yfinance==0.2.51
# pandas==2.2.3
# numpy==2.1.3
# requests==2.32.3
# apscheduler==3.10.4
# pytz==2024.2
# ─────────────────────────────────────────────────────────────────────────

import os
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import anthropic
import uvicorn
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pytz

# ─── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─── Variables de entorno ─────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("❌ Falta ANTHROPIC_API_KEY en variables de entorno")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("❌ Falta TELEGRAM_BOT_TOKEN en variables de entorno")
if not TELEGRAM_CHAT_ID:
    raise RuntimeError("❌ Falta TELEGRAM_CHAT_ID en variables de entorno")

# ─── Clientes ─────────────────────────────────────────────────────────────
ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ─── Configuración de símbolos ────────────────────────────────────────────
# yfinance usa tickers continuos de futuros
SYMBOLS = {
    "MNQ": {
        "ticker":     "MNQ=F",
        "name":       "Micro Nasdaq-100",
        "tick_size":  0.25,
        "tick_value": 0.50,
        "point_value": 2.00,   # $2 por punto
    },
    "MES": {
        "ticker":     "MES=F",
        "name":       "Micro S&P 500",
        "tick_size":  0.25,
        "tick_value": 1.25,
        "point_value": 5.00,   # $5 por punto
    },
    "MYM": {
        "ticker":     "MYM=F",
        "name":       "Micro Dow Jones",
        "tick_size":  1.00,
        "tick_value": 0.50,
        "point_value": 0.50,   # $0.50 por punto
    },
}

# ─── Gestión de riesgo ────────────────────────────────────────────────────
MAX_RISK_USD = 250  # Riesgo máximo por operación en dólares

def calc_position_size(symbol: str, risk_points: float) -> dict:
    """
    Calcula el número de contratos basado en riesgo fijo de $250.

    Fórmula: Contratos = MAX_RISK_USD / (risk_points × point_value)

    Valores por punto:
      MNQ → $2.00 por punto
      MES → $5.00 por punto
      MYM → $0.50 por punto

    Devuelve dict con: contracts (int), risk_per_contract (float),
                       total_risk (float), point_value (float)
    """
    point_value   = SYMBOLS[symbol]["point_value"]
    risk_per_cont = risk_points * point_value          # $ en riesgo por contrato

    if risk_per_cont <= 0:
        return {
            "contracts":        1,
            "risk_per_contract": 0,
            "total_risk":       0,
            "point_value":      point_value,
            "note":             "Riesgo por contrato inválido, usando 1 contrato mínimo",
        }

    contracts = int(MAX_RISK_USD / risk_per_cont)      # redondea HACIA ABAJO siempre
    contracts = max(1, contracts)                       # mínimo 1 contrato

    total_risk = contracts * risk_per_cont             # riesgo real con ese número

    return {
        "contracts":         contracts,
        "risk_per_contract": round(risk_per_cont, 2),
        "total_risk":        round(total_risk, 2),
        "point_value":       point_value,
    }

# ─── Estado global (en memoria) ───────────────────────────────────────────
last_signals: dict[str, dict] = {}   # Última señal enviada por símbolo
signal_history: list[dict]    = []   # Historial de las últimas 100 señales
ema_state: dict[str, dict]    = {}   # Estado EMA anterior para detectar cruces

ET = pytz.timezone("America/New_York")

# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — DATOS DE MERCADO
# ═══════════════════════════════════════════════════════════════════════════

def get_market_data(ticker: str, lookback_bars: int = 250) -> Optional[pd.DataFrame]:
    """
    Descarga datos OHLCV de yfinance en timeframe de 5 minutos.
    Devuelve DataFrame con columnas: open, high, low, close, volume
    """
    try:
        # yfinance: "5m" disponible para los últimos 60 días
        data = yf.download(
            tickers=ticker,
            period="5d",        # últimos 5 días (suficiente para EMA200 en 5m)
            interval="5m",
            progress=False,
            auto_adjust=True,
        )
        if data is None or data.empty:
            log.warning(f"Sin datos para {ticker}")
            return None

        # Normalizar columnas
        data.columns = [c.lower() for c in data.columns]
        data = data[["open", "high", "low", "close", "volume"]].copy()
        data.dropna(inplace=True)

        if len(data) < 210:
            log.warning(f"{ticker}: solo {len(data)} barras (necesita ≥210 para EMA200)")
            return None

        return data

    except Exception as e:
        log.error(f"Error descargando {ticker}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — CÁLCULO DE INDICADORES
# ═══════════════════════════════════════════════════════════════════════════

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP diario (se resetea cada sesión RTH)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol  = typical * df["volume"]
    # Agrupar por fecha para resetear VWAP diario
    dates = df.index.normalize()
    vwap  = pd.Series(index=df.index, dtype=float)
    for d in dates.unique():
        mask        = dates == d
        vwap[mask]  = tp_vol[mask].cumsum() / df["volume"][mask].cumsum()
    return vwap

def calc_volume_ma(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(window=period).mean()

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([
        h - l,
        (h - c).abs(),
        (l - c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema9"]    = calc_ema(df["close"], 9)
    df["ema20"]   = calc_ema(df["close"], 20)
    df["ema200"]  = calc_ema(df["close"], 200)
    df["vwap"]    = calc_vwap(df)
    df["vol_ma20"]= calc_volume_ma(df["volume"], 20)
    df["atr14"]   = calc_atr(df)
    df["vol_ratio"]= df["volume"] / df["vol_ma20"]  # >1.5 = volumen institucional
    return df


# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — DETECCIÓN DE SEÑAL (LÓGICA ESTRATEGIA)
# ═══════════════════════════════════════════════════════════════════════════

def detect_signal(symbol: str, df: pd.DataFrame) -> Optional[dict]:
    """
    Estrategia:
    ─ LONG:  EMA9 cruza por ENCIMA de EMA20 + precio > EMA200
             + pullback hacia EMA9/EMA20 después del cruce
             + precio ≥ VWAP + volumen institucional (ratio ≥ 1.3)

    ─ SHORT: EMA9 cruza por DEBAJO de EMA20 + precio < EMA200
             + pullback hacia EMA9/EMA20 después del cruce
             + precio ≤ VWAP + volumen institucional (ratio ≥ 1.3)

    Devuelve dict con los datos del setup o None si no hay señal.
    """

    if len(df) < 5:
        return None

    # ── Últimas 3 barras (índice -3, -2, -1) ──────────────────────────────
    # -3 y -2 para detectar el cruce, -1 es la barra actual (pullback)
    prev2 = df.iloc[-3]   # 2 barras atrás
    prev1 = df.iloc[-2]   # barra del cruce
    curr  = df.iloc[-1]   # barra actual (pullback / entrada)

    # ── Valores actuales ───────────────────────────────────────────────────
    price    = float(curr["close"])
    ema9     = float(curr["ema9"])
    ema20    = float(curr["ema20"])
    ema200   = float(curr["ema200"])
    vwap     = float(curr["vwap"])
    atr      = float(curr["atr14"])
    vol_ratio= float(curr["vol_ratio"]) if not np.isnan(curr["vol_ratio"]) else 0

    # ── Detección de cruce (en barra -2, la barra anterior a la actual) ───
    cross_up   = float(prev2["ema9"]) < float(prev2["ema20"]) and \
                 float(prev1["ema9"]) > float(prev1["ema20"])

    cross_down = float(prev2["ema9"]) > float(prev2["ema20"]) and \
                 float(prev1["ema9"]) < float(prev1["ema20"])

    # ── Condiciones LONG ───────────────────────────────────────────────────
    if cross_up:
        above_ema200 = price > ema200
        pullback     = curr["low"] <= max(ema9, ema20) * 1.001  # toca zona EMA
        near_vwap_or_above = price >= vwap * 0.999
        inst_volume  = vol_ratio >= 1.3

        conditions = {
            "Precio sobre EMA200": above_ema200,
            "Pullback a EMA9/20":  pullback,
            "Precio ≥ VWAP":       near_vwap_or_above,
            "Volumen institucional": inst_volume,
        }
        passed = sum(conditions.values())

        if passed >= 3:  # al menos 3 de 4 condiciones
            sl       = round(min(ema9, ema20) - atr * 0.5, 2)
            rr2      = round(price + (price - sl) * 2, 2)   # R/R 1:2
            rr3      = round(price + (price - sl) * 3, 2)   # R/R 1:3
            conf     = "ALTA" if passed == 4 else "MEDIA"
            risk_pts = round(price - sl, 2)
            sizing   = calc_position_size(symbol, risk_pts)

            return {
                "direction":         "LONG 🟢",
                "direction_raw":     "LONG",
                "symbol":            symbol,
                "price":             price,
                "entry":             price,
                "stop_loss":         sl,
                "target_1":          rr2,
                "target_2":          rr3,
                "risk_points":       risk_pts,
                "confidence":        conf,
                "ema9":              round(ema9, 2),
                "ema20":             round(ema20, 2),
                "ema200":            round(ema200, 2),
                "vwap":              round(vwap, 2),
                "vol_ratio":         round(vol_ratio, 2),
                "atr":               round(atr, 2),
                "conditions":        conditions,
                "conditions_passed": passed,
                "cross_bar":         prev1.name.isoformat() if hasattr(prev1.name, "isoformat") else str(prev1.name),
                "contracts":         sizing["contracts"],
                "risk_per_contract": sizing["risk_per_contract"],
                "total_risk":        sizing["total_risk"],
                "point_value":       sizing["point_value"],
            }

    # ── Condiciones SHORT ──────────────────────────────────────────────────
    if cross_down:
        below_ema200 = price < ema200
        pullback     = curr["high"] >= min(ema9, ema20) * 0.999
        near_vwap_or_below = price <= vwap * 1.001
        inst_volume  = vol_ratio >= 1.3

        conditions = {
            "Precio bajo EMA200":   below_ema200,
            "Pullback a EMA9/20":   pullback,
            "Precio ≤ VWAP":        near_vwap_or_below,
            "Volumen institucional": inst_volume,
        }
        passed = sum(conditions.values())

        if passed >= 3:
            sl       = round(max(ema9, ema20) + atr * 0.5, 2)
            rr2      = round(price - (sl - price) * 2, 2)
            rr3      = round(price - (sl - price) * 3, 2)
            conf     = "ALTA" if passed == 4 else "MEDIA"
            risk_pts = round(sl - price, 2)
            sizing   = calc_position_size(symbol, risk_pts)

            return {
                "direction":         "SHORT 🔴",
                "direction_raw":     "SHORT",
                "symbol":            symbol,
                "price":             price,
                "entry":             price,
                "stop_loss":         sl,
                "target_1":          rr2,
                "target_2":          rr3,
                "risk_points":       risk_pts,
                "confidence":        conf,
                "ema9":              round(ema9, 2),
                "ema20":             round(ema20, 2),
                "ema200":            round(ema200, 2),
                "vwap":              round(vwap, 2),
                "vol_ratio":         round(vol_ratio, 2),
                "atr":               round(atr, 2),
                "conditions":        conditions,
                "conditions_passed": passed,
                "cross_bar":         prev1.name.isoformat() if hasattr(prev1.name, "isoformat") else str(prev1.name),
                "contracts":         sizing["contracts"],
                "risk_per_contract": sizing["risk_per_contract"],
                "total_risk":        sizing["total_risk"],
                "point_value":       sizing["point_value"],
            }

    return None  # Sin señal


# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — ANÁLISIS CON CLAUDE AI
# ═══════════════════════════════════════════════════════════════════════════

def analyze_with_claude(setup: dict) -> dict:
    """
    Envía el setup detectado a Claude para validación adicional
    y redacción del mensaje de señal.
    Devuelve dict con: valid (bool), message (str), extra_notes (str)
    """
    spec = SYMBOLS[setup["symbol"]]
    cond_text = "\n".join(
        f"  {'✅' if v else '❌'} {k}"
        for k, v in setup["conditions"].items()
    )

    prompt = f"""Eres un trader profesional especializado en futuros micro del CME.
Valida el siguiente setup de trading y redacta el mensaje de señal para Telegram.

═══════════════════════════════════════
SETUP DETECTADO
Contrato : {setup['symbol']} — {spec['name']}
Dirección: {setup['direction']}
Confianza: {setup['confidence']}
Precio   : {setup['price']}
─── Niveles ───────────────────────────
Entrada  : {setup['entry']}
Stop Loss: {setup['stop_loss']} ({setup['risk_points']} puntos de riesgo)
Target 1 : {setup['target_1']} (R/R 1:2)
Target 2 : {setup['target_2']} (R/R 1:3)
─── Tamaño de posición ($250 riesgo máx) ──
Contratos: {setup['contracts']}
Riesgo/contrato: ${setup['risk_per_contract']}
Riesgo total   : ${setup['total_risk']}
Valor por punto: ${setup['point_value']}
─── Indicadores ───────────────────────
EMA 9    : {setup['ema9']}
EMA 20   : {setup['ema20']}
EMA 200  : {setup['ema200']}
VWAP     : {setup['vwap']}
Vol Ratio: {setup['vol_ratio']}x promedio
ATR 14   : {setup['atr']}
─── Condiciones ({setup['conditions_passed']}/4) ─────────
{cond_text}
Cruce detectado en barra: {setup['cross_bar']}
═══════════════════════════════════════

TAREA:
1. Evalúa si el setup es válido o tiene debilidades importantes.
2. Si es válido, redacta el mensaje de Telegram (máximo 280 palabras).
3. El mensaje debe ser claro, profesional y en español.
4. Incluye emoji relevantes para mejor lectura en móvil.
5. El mensaje DEBE incluir la sección de contratos con riesgo total real.

Responde ÚNICAMENTE con este JSON (sin markdown, sin texto extra):
{{
  "valid": true,
  "confidence_final": "ALTA|MEDIA|BAJA",
  "telegram_message": "... mensaje completo para Telegram ...",
  "extra_notes": "... nota interna opcional para logs ..."
}}

Si el setup tiene fallas críticas (stop demasiado amplio, contra tendencia mayor, etc.),
pon "valid": false y explica en "extra_notes" por qué se descarta."""

    log.info(f"📤 Enviando setup {setup['symbol']} {setup['direction_raw']} a Claude...")

    try:
        msg = ai_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()

        # Limpiar posibles backticks
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        log.info(f"📥 Claude: valid={result.get('valid')} conf={result.get('confidence_final')}")
        return result

    except json.JSONDecodeError as e:
        log.error(f"JSON parse error de Claude: {e}")
        return {"valid": False, "extra_notes": f"JSON inválido: {e}"}
    except Exception as e:
        log.error(f"Error Claude API: {e}")
        return {"valid": False, "extra_notes": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════

def send_telegram(message: str) -> bool:
    """Envía mensaje de texto a Telegram vía Bot API."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            log.info("✅ Mensaje enviado a Telegram")
            return True
        else:
            log.error(f"Telegram error {r.status_code}: {r.text}")
            return False
    except Exception as e:
        log.error(f"Error enviando Telegram: {e}")
        return False

def send_startup_message():
    """Notifica en Telegram que el agente está activo."""
    now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    msg = (
        "🤖 <b>Agente de Señales Futuros ACTIVADO</b>\n\n"
        f"📅 {now}\n"
        "📊 Símbolos: <b>MNQ · MES · MYM</b>\n"
        "⏱ Análisis: cada 5 minutos\n"
        "📈 Estrategia: EMA 9/20/200 + Pullback + VWAP + Volumen\n\n"
        "✅ Esperando setups de alta probabilidad...\n"
        "<i>Solo se envían señales cuando TODAS las condiciones están alineadas.</i>"
    )
    send_telegram(msg)


# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — LOOP PRINCIPAL DE ANÁLISIS (cada 5 min)
# ═══════════════════════════════════════════════════════════════════════════

def is_market_hours() -> bool:
    """
    Futuros micro operan casi 24/5.
    Filtramos el cierre de mantenimiento diario 5:00-6:00 PM ET.
    """
    now = datetime.now(ET)
    # Fin de semana
    if now.weekday() >= 5:
        return False
    h, m = now.hour, now.minute
    # Mantenimiento 17:00 - 18:00 ET
    if h == 17 or (h == 16 and m >= 55):
        return False
    return True

def avoid_duplicate(symbol: str, direction: str) -> bool:
    """
    Evita enviar la misma señal dos veces seguidas.
    Cooldown: no repetir la misma dirección en 30 minutos.
    """
    if symbol not in last_signals:
        return False
    last = last_signals[symbol]
    if last.get("direction_raw") != direction:
        return False
    elapsed = (datetime.now(timezone.utc) - last.get("sent_at", datetime.min.replace(tzinfo=timezone.utc))).total_seconds()
    return elapsed < 1800  # 30 minutos

async def analyze_symbol(symbol: str):
    """Analiza un símbolo y envía señal si el setup es válido."""
    spec = SYMBOLS[symbol]
    log.info(f"🔍 Analizando {symbol} ({spec['ticker']})...")

    # 1. Obtener datos
    df = get_market_data(spec["ticker"])
    if df is None:
        log.warning(f"{symbol}: sin datos, saltando")
        return

    # 2. Calcular indicadores
    df = add_indicators(df)

    # 3. Detectar setup
    setup = detect_signal(symbol, df)
    if setup is None:
        log.info(f"{symbol}: sin setup en este momento")
        return

    log.info(f"{symbol}: setup detectado → {setup['direction']} [{setup['confidence']}]")

    # 4. Evitar duplicados
    if avoid_duplicate(symbol, setup["direction_raw"]):
        log.info(f"{symbol}: señal duplicada, ignorando (cooldown 30min)")
        return

    # 5. Validar con Claude
    claude_result = analyze_with_claude(setup)

    if not claude_result.get("valid", False):
        log.info(f"{symbol}: Claude descartó el setup — {claude_result.get('extra_notes', '')}")
        return

    # 6. Construir mensaje final
    telegram_msg = claude_result.get("telegram_message", "")

    # Si Claude no generó mensaje, construir uno de respaldo
    if not telegram_msg:
        now_et = datetime.now(ET).strftime("%H:%M ET")
        telegram_msg = (
            f"📡 <b>SEÑAL {setup['direction_raw']}</b> — {setup['symbol']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕐 {now_et}\n"
            f"💰 Entrada  : <b>{setup['entry']}</b>\n"
            f"🛑 Stop Loss: <b>{setup['stop_loss']}</b>\n"
            f"🎯 Target 1 : <b>{setup['target_1']}</b> (R/R 1:2)\n"
            f"🎯 Target 2 : <b>{setup['target_2']}</b> (R/R 1:3)\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📦 <b>TAMAÑO DE POSICIÓN</b>\n"
            f"   Contratos : <b>{setup['contracts']}</b>\n"
            f"   Riesgo/cto: ${setup['risk_per_contract']:.2f}  "
            f"(${setup['point_value']}/pto × {setup['risk_points']} ptos)\n"
            f"   Riesgo total: <b>${setup['total_risk']:.2f}</b> / máx $250\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 EMA9: {setup['ema9']} | EMA20: {setup['ema20']}\n"
            f"📊 EMA200: {setup['ema200']} | VWAP: {setup['vwap']}\n"
            f"📊 Vol: {setup['vol_ratio']}x | ATR: {setup['atr']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"⭐ Confianza: <b>{claude_result.get('confidence_final', setup['confidence'])}</b>\n"
            f"⚠️ <i>Solo educativo. Gestiona tu riesgo.</i>"
        )

    # 7. Enviar a Telegram
    sent = send_telegram(telegram_msg)

    if sent:
        # Guardar estado
        last_signals[symbol] = {
            **setup,
            "sent_at": datetime.now(timezone.utc),
            "claude_confidence": claude_result.get("confidence_final"),
        }
        signal_history.append({
            **setup,
            "sent_at":           datetime.now(timezone.utc).isoformat(),
            "claude_confidence": claude_result.get("confidence_final"),
            "telegram_sent":     True,
        })
        if len(signal_history) > 100:
            signal_history.pop(0)

        log.info(f"✅ Señal enviada: {symbol} {setup['direction']} @ {setup['entry']}")

async def run_analysis():
    """Corre el análisis de todos los símbolos."""
    if not is_market_hours():
        log.info("⏸ Mercado cerrado o mantenimiento — esperando...")
        return

    log.info("━━━━━━━ Iniciando análisis de mercado ━━━━━━━")
    for symbol in SYMBOLS:
        try:
            await analyze_symbol(symbol)
        except Exception as e:
            log.error(f"Error inesperado analizando {symbol}: {e}")
    log.info("━━━━━━━ Análisis completado ━━━━━━━\n")


# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — FASTAPI (Railway necesita un servidor HTTP activo)
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Agente Señales Futuros",
    description="MNQ · MES · MYM — EMA 9/20/200 + Claude AI + Telegram",
    version="2.0.0"
)

@app.get("/")
def root():
    market_open = is_market_hours()
    return {
        "status":       "🟢 Agente activo",
        "version":      "2.0.0",
        "market_open":  market_open,
        "symbols":      list(SYMBOLS.keys()),
        "strategy":     "EMA 9/20/200 + Pullback + VWAP + Volumen Institucional",
        "interval":     "5 minutos",
        "last_signals": {
            sym: {
                "direction": v.get("direction"),
                "entry":     v.get("entry"),
                "sent_at":   v.get("sent_at", "").isoformat()
                             if hasattr(v.get("sent_at", ""), "isoformat")
                             else str(v.get("sent_at", "")),
            }
            for sym, v in last_signals.items()
        }
    }

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "timestamp": datetime.now(ET).isoformat(),
        "market":    "open" if is_market_hours() else "closed",
        "signals_sent_today": len([
            s for s in signal_history
            if s.get("sent_at", "")[:10] == datetime.now(ET).strftime("%Y-%m-%d")
        ])
    }

@app.get("/history")
def history(symbol: Optional[str] = None, limit: int = 20):
    filtered = signal_history
    if symbol:
        filtered = [s for s in signal_history if s["symbol"] == symbol.upper()]
    return {
        "total":   len(filtered),
        "signals": list(reversed(filtered[-limit:]))
    }

@app.get("/scan")
async def manual_scan():
    """Fuerza un análisis inmediato (útil para testing)."""
    await run_analysis()
    return {"status": "Análisis completado", "check": "/history para ver resultados"}


# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 — STARTUP Y SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    log.info("🚀 Iniciando Agente de Señales Futuros v2.0")

    # Notificar en Telegram que el bot está activo
    send_startup_message()

    # Scheduler: analiza cada 5 minutos
    scheduler = AsyncIOScheduler(timezone=str(ET))
    scheduler.add_job(
        run_analysis,
        trigger="cron",
        minute="*/5",   # cada 5 minutos exactos
        id="market_analysis",
        name="Análisis de señales"
    )
    scheduler.start()
    log.info("⏱ Scheduler activo — análisis cada 5 minutos")

    # Primera ejecución inmediata al arrancar
    asyncio.create_task(run_analysis())


# ─── Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
