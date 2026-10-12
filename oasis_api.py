"""
绿洲大模型 - PCSE/WOFOST 作物模拟 API (v2.0 - 接入真实土壤/墒情/GIS)
==================================================================
- /simulate: 动态获取 SoilGrids 真实土壤参数 + NASA POWER 气象
- /soil: 土壤属性查询
- /soil-moisture: 墒情时间序列 (NASA POWER GWET)
- /gis/elevation: 高程查询
- /gis/boundary: 行政区划 GeoJSON
"""
import os
import sys
import traceback
import time
import tempfile
import sqlite3
import json
import urllib.request
import threading
import asyncio
from contextlib import asynccontextmanager
from collections import namedtuple
from datetime import datetime, timedelta, date
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

# ★ 启动时后台预热常用坐标的土壤缓存（土壤是静态数据，取一次长期复用；
#    配合外部保活让实例常驻，用户提问时基本都是缓存命中、秒回）
SOIL_WARM_COORDS = [(38.48, 77.26), (41.17, 80.25), (44.30, 86.03), (43.92, 81.32), (43.82, 87.62)]

@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_soil_warm_forever, daemon=True).start()
    yield

app = FastAPI(title="绿洲大模型 - 作物模拟 API", version="2.1.0", lifespan=lifespan)

# ========== API Key 保护 ==========
# 设置 OASIS_API_KEY 环境变量后，所有请求需带 X-API-Key 头
OASIS_API_KEY = os.environ.get("OASIS_API_KEY", "")

@app.middleware("http")
async def api_key_middleware(request, call_next):
    if OASIS_API_KEY:
        key = request.headers.get("X-API-Key", "")
        if key != OASIS_API_KEY:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "无效的 API Key"}, status_code=401)
    return await call_next(request)

# ---------- 启动时初始化数据库 ----------
print(f"[启动] HOME: {os.environ.get('HOME', 'N/A')}")
db_path = os.path.expanduser("~/.pcse/pcse.db")
print(f"[启动] 数据库路径: {db_path}")

if not os.path.exists(db_path) or os.path.getsize(db_path) < 1000:
    print("[启动] 数据库不完整，正在初始化...")
    if os.path.exists(db_path):
        os.remove(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    import pcse
    from pcse.util import load_SQLite_dump_file
    pcse_home = os.path.dirname(pcse.__file__)
    dump_file = os.path.join(pcse_home, "tests", "test_data", "pcse_dump.sql")
    if os.path.exists(dump_file):
        load_SQLite_dump_file(dump_file, db_path)
        print(f"[启动] 数据库初始化完成")
    else:
        print(f"[启动] 找不到建库文件: {dump_file}")

import pcse
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
n = cur.fetchone()[0]
print(f"[启动] PCSE {pcse.__version__}, 数据库 {n} 个表")
conn.close()

# ============================================================
# 数据模型
# ============================================================
class SimRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, example=38.48)
    longitude: float = Field(..., ge=-180, le=180, example=77.26)
    crop_no: int = Field(default=1, ge=1, le=17, description="1=冬小麦 2=玉米 7=马铃薯")
    planting_date: str = Field(..., example="2025-04-01")
    production_mode: str = Field(default="wlp", pattern="^(pp|wlp)$")
    initial_soil_moisture: float = Field(default=100.0, ge=0)

class SimResult(BaseModel):
    status: str = "success"
    duration_seconds: float
    summary: dict
    timeseries: dict
    soil_info: Optional[dict] = None
    message: str = ""
    cached: bool = False  # 是否命中缓存

# ============================================================
# 外部数据接入: SoilGrids 土壤
# ============================================================
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

def http_get_json(url, timeout=30, retries=3):
    """带重试的 HTTP GET"""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt * 2)
    raise last_err

def _daily_val(daily: dict, key: str, i: int):
    """安全取 daily 数组第 i 个值（key 缺失或数组比 time 短时返回 None，防 IndexError）"""
    arr = daily.get(key) or []
    return arr[i] if i < len(arr) else None

def fetch_soilgrids(lat, lon, timeout=45, retries=1):
    """从 ISRIC SoilGrids 获取土壤属性剖面"""
    url = (f"{SOILGRIDS_URL}?lon={lon}&lat={lat}"
           f"&property=clay&property=sand&property=silt&property=bdod&property=soc"
           f"&depth=0-5cm&depth=5-15cm&depth=15-30cm&depth=30-60cm&depth=60-100cm"
           f"&value=mean")
    # ★ ISRIC 实测 23~90s 才返回（波动极大），默认 30s×3 次常常整轮白跑
    return http_get_json(url, timeout=timeout, retries=retries)

def _depth_value(layer, depths):
    """按优先级取深度层值"""
    for d in depths:
        for dep in layer["depths"]:
            if dep["label"] == d and dep["values"].get("mean") is not None:
                return dep["values"]["mean"]
    return None

def saxton_rawls(sand_frac, clay_frac, om_frac):
    """Saxton & Rawls (2006) pedotransfer: 砂/黏/有机质 → 土壤水分常数"""
    sm0 = (0.278 * sand_frac + 0.034 * clay_frac + 0.022 * om_frac
           - 0.018 * sand_frac * om_frac - 0.027 * clay_frac * om_frac
           + 0.584 * sand_frac * clay_frac + 0.078)
    smfcf = (-0.251 * sand_frac + 0.195 * clay_frac + 0.011 * om_frac
             + 0.006 * sand_frac * om_frac - 0.027 * clay_frac * om_frac
             + 0.452 * sand_frac * clay_frac + 0.299)
    smw = (-0.024 * sand_frac + 0.487 * clay_frac + 0.006 * om_frac
           + 0.005 * sand_frac * om_frac - 0.013 * clay_frac * om_frac
           + 0.068 * sand_frac * clay_frac + 0.031)
    return sm0, smfcf, smw

def texture_class(sand, clay):
    if sand >= 70 and clay <= 15: return "砂土"
    if sand >= 55 and clay <= 20: return "砂质壤土"
    if sand >= 40 and clay <= 27: return "壤土"
    if clay >= 35: return "黏土"
    return "壤土"

# ★ 土壤属性缓存：同一坐标只查一次真实数据源（土壤是静态数据，进程内永久有效；重启即重建）
_soil_cache = {}
_soil_retry = set()
_soil_lock = threading.Lock()


def _soil_retry_add(key):
    with _soil_lock:
        if key not in _soil_cache:
            _soil_retry.add(key)


def _soil_retry_take():
    with _soil_lock:
        keys = [k for k in _soil_retry if k not in _soil_cache]
        _soil_retry.clear()
        return keys


def _soil_warm_forever():
    """后台守护线程：不限期把常用坐标 + 用户查过但失败的坐标补齐进缓存。
    ISRIC 源很慢（50~90s，还会超时），只能靠"提前取好 + 命中缓存"保证用户体验。
    只对 5 个常用坐标 + 实际被问到的坐标取数，不主动扫点。"""
    time.sleep(10)
    round_no = 0
    while True:
        round_no += 1
        todo = []
        with _soil_lock:
            for lat, lon in SOIL_WARM_COORDS:
                k = (round(float(lat), 3), round(float(lon), 3))
                if k not in _soil_cache:
                    todo.append(k)
            for k in list(_soil_retry):
                if k not in _soil_cache and k not in todo:
                    todo.append(k)
        for lat, lon in todo:
            try:
                soilgrids_to_soildata(lat, lon, timeout=90, retries=3)
                print(f"[土壤预热] 第{round_no}轮 已缓存 {lat},{lon}")
            except Exception as e:
                print(f"[土壤预热] 第{round_no}轮 {lat},{lon} 暂不可得: {type(e).__name__}")
            time.sleep(5)
        with _soil_lock:
            cached = len(_soil_cache)
        print(f"[土壤预热] 第{round_no}轮结束，缓存坐标 {cached} 个，待补 {len(todo)} 个；15 分钟后再查")
        time.sleep(900)

def soilgrids_to_soildata(lat, lon, timeout=45, retries=1):
    """SoilGrids 数据 → WOFOST 土壤参数 (soildata dict)。带坐标缓存，命中即秒回。"""
    try:
        key = (round(float(lat), 3), round(float(lon), 3))
    except Exception:
        key = None
    if key is not None:
        with _soil_lock:
            hit = _soil_cache.get(key)
        if hit is not None:
            sd, info = hit
            info = dict(info)
            info["cached"] = True
            return sd, info
    depths = ["30-60cm", "15-30cm", "5-15cm", "0-5cm"]
    try:
        data = fetch_soilgrids(lat, lon, timeout=timeout, retries=retries)
    except Exception:
        if key is not None:
            _soil_retry_add(key)      # ★ 失败进后台补取队列，不限期重试直到缓存成功
        raise
    props = {layer["name"]: layer for layer in data["properties"]["layers"]}

    def safe(layer, div):
        v = _depth_value(layer, depths)
        return v / div if v is not None else None

    sand = safe(props["sand"], 10.0)
    clay = safe(props["clay"], 10.0)
    silt = safe(props["silt"], 10.0)
    soc = safe(props["soc"], 10.0)
    bdod = safe(props["bdod"], 100.0)

    # 兜底默认值（参考壤土）
    DEFAULTS = {"sand": 35.0, "clay": 20.0, "silt": 45.0, "soc": 8.0, "bdod": 1.40}
    used_default = False
    for k, v in [("sand", sand), ("clay", clay), ("silt", silt), ("soc", soc), ("bdod", bdod)]:
        if v is None:
            used_default = True
    if sand is None: sand = DEFAULTS["sand"]
    if clay is None: clay = DEFAULTS["clay"]
    if silt is None: silt = DEFAULTS["silt"]
    if soc is None: soc = DEFAULTS["soc"]
    if bdod is None: bdod = DEFAULTS["bdod"]

    om_frac = (soc * 1.72) / 1000.0
    sm0, smfcf, smw = saxton_rawls(sand / 100.0, clay / 100.0, om_frac)

    # 饱和导水率按砂含量粗估 (cm/day)
    if sand >= 70: k0 = 300.0
    elif sand >= 55: k0 = 150.0
    elif sand >= 40: k0 = 60.0
    elif sand >= 25: k0 = 25.0
    else: k0 = 10.0

    soildata = {
        "RDMSOL": 60.0,
        "CRAIRC": 0.06,
        "K0": k0,
        "SOPE": k0 * 0.7,
        "KSUB": k0 * 0.5,
        "SMFCF": round(smfcf, 4),
        "SM0": round(sm0, 4),
        "SMW": round(smw, 4),
    }
    soil_info = {
        "soil_texture": texture_class(sand, clay),
        "sand_pct": round(sand, 1),
        "clay_pct": round(clay, 1),
        "silt_pct": round(silt, 1),
        "organic_carbon_gkg": round(soc, 1),
        "bulk_density_gcm3": round(bdod, 2),
        "used_default": used_default,
        "data_source": "ISRIC SoilGrids",
    }
    if key is not None and not used_default:
        # 只缓存真实取到的数据；回退默认值的不缓存，下次还能再试
        with _soil_lock:
            _soil_cache[key] = (soildata, dict(soil_info))
    return soildata, soil_info

# ============================================================
# 外部数据接入: NASA POWER 墒情 (GWETTOP/GWETPROF)
# ============================================================
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

def fetch_soil_moisture(lat, lon, start=None, end=None):
    """从 NASA POWER 获取土壤墒情时间序列"""
    start = _norm_date(start, (date.today() - timedelta(days=30)).strftime("%Y%m%d"))
    end = _norm_date(end, date.today().strftime("%Y%m%d"))
    if start > end:
        raise HTTPException(400, f"日期范围无效: start={start} > end={end}")
    url = (f"{POWER_URL}?parameters=GWETTOP,GWETPROF&community=AG"
           f"&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON")
    data = http_get_json(url, timeout=45)
    params = data["properties"]["parameter"]
    result = {
        "location": {"lat": lat, "lon": lon},
        "period": {"start": start, "end": end},
        "source": "NASA POWER (MERRA-2 再分析)",
        "series": [],
    }
    gwet_top = params.get("GWETTOP", {})
    gwet_prof = params.get("GWETPROF", {})
    for day in sorted(gwet_top.keys()):
        result["series"].append({
            "date": f"{day[:4]}-{day[4:6]}-{day[6:8]}",
            "top_0_5cm": gwet_top.get(day),
            "rootzone_0_100cm": gwet_prof.get(day),
        })
    # 汇总
    tops = [s["top_0_5cm"] for s in result["series"] if s["top_0_5cm"] is not None]
    if tops:
        result["summary"] = {
            "top_mean": round(sum(tops) / len(tops), 3),
            "top_min": round(min(tops), 3),
            "top_max": round(max(tops), 3),
        }
    return result

# ============================================================
# 外部数据接入: GIS
# ============================================================
def fetch_elevation(lat, lon):
    """高程查询: Open-Elevation (免key), 失败回退 NASA POWER"""
    try:
        url = "https://api.open-elevation.com/api/v1/lookup"
        req = urllib.request.Request(
            url,
            data=json.dumps({"locations": [{"latitude": lat, "longitude": lon}]}).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            r = json.loads(resp.read().decode())
            return {"elevation_m": round(r["results"][0]["elevation"], 1), "source": "Open-Elevation"}
    except Exception:
        # 回退: NASA POWER 自带海拔
        try:
            url = (f"{POWER_URL}?parameters=T2M&community=AG&longitude={lon}"
                   f"&latitude={lat}&start={date.today().strftime('%Y%m%d')}"
                   f"&end={date.today().strftime('%Y%m%d')}&format=JSON")
            data = http_get_json(url, timeout=30)
            elev = data["geometry"]["coordinates"][2]
            return {"elevation_m": round(elev, 1), "source": "NASA POWER"}
        except Exception:
            return {"elevation_m": None, "source": "unavailable"}

def fetch_boundary(code):
    """行政区划 GeoJSON (阿里云 DataV)"""
    url = f"https://geo.datav.aliyun.com/areas_v3/bound/{code}_full.json"
    try:
        return http_get_json(url, timeout=20)
    except Exception as e:
        raise HTTPException(404, f"未找到行政区划 {code}: {e}")

# ============================================================
# 从演示数据库加载作物参数
# ============================================================
def load_params(crop_no, wav=100.0, soildata=None, grid=31031):
    """从 PCSE 演示数据库加载作物参数; 土壤参数可用外部数据覆盖"""
    from pcse.tests.db_input import fetch_cropdata, fetch_sitedata, fetch_soildata
    from pcse.base import ParameterProvider

    conn = sqlite3.connect(db_path)
    conn.row_factory = lambda c, r: namedtuple("Row", [d[0] for d in c.description])._make(r)

    try:
        sited = fetch_sitedata(conn, grid, 2000)
        sited["WAV"] = wav
        cropd = fetch_cropdata(conn, grid, 2000, crop_no)
        if soildata is None:
            soild = fetch_soildata(conn, grid)
        else:
            soild = soildata
        params = ParameterProvider(sitedata=sited, cropdata=cropd, soildata=soild)
        # 获取作物名
        cur = conn.cursor()
        cur.execute("SELECT crop_name FROM crop WHERE crop_no=?", (crop_no,))
        row = cur.fetchone()
        crop_name = row[0] if row else f"crop_{crop_no}"
        conn.close()
        return params, crop_name
    except Exception as e:
        conn.close()
        raise e

# ---------- 农事管理 YAML ----------
def build_agro_yaml(plant_date, crop_name):
    pd_ = datetime.strptime(plant_date, "%Y-%m-%d").date()
    harvest = pd_ + timedelta(days=200)
    lines = [
        "AgroManagement:",
        f"- {pd_}:",
        "    CropCalendar:",
        f"        crop_name: {crop_name.replace(' ', '_')}",
        "        variety_name: '1'",
        f"        crop_start_date: {pd_}",
        "        crop_start_type: emergence",
        f"        crop_end_date: {harvest}",
        "        crop_end_type: maturity",
        "        max_duration: 365",
        "    TimedEvents: null",
        "    StateEvents: null",
    ]
    return "\n".join(lines)

# ============================================================
# API 端点
# ============================================================

@app.get("/")
def root():
    """服务状态 + 可用作物列表"""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT crop_no, crop_name FROM crop ORDER BY crop_no")
        crops = {str(r[0]): r[1] for r in cur.fetchall()}
        conn.close()
    except Exception as e:
        crops = {"error": str(e)}
    return {
        "service": "绿洲大模型 - 作物模拟 API",
        "version": "2.0.0",
        "status": "online",
        "available_crops": crops,
        "models": ["pp (potential production)", "wlp (water-limited production)"],
        "data_sources": {
            "weather": "NASA POWER (气象)",
            "soil": "ISRIC SoilGrids (真实土壤)",
            "soil_moisture": "NASA POWER GWET (墒情)",
            "elevation": "Open-Elevation (高程)",
            "boundary": "DataV GeoAtlas (行政区划)",
        },
    }

def _check_coords(lat: float, lon: float):
    """校验经纬度范围，非法时返回 400（避免打到外部 API 才报错）"""
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(400, f"无效坐标: lat={lat}, lon={lon}（lat 应在 -90~90，lon 应在 -180~180）")

def _norm_date(s, default):
    """日期归一化为 YYYYMMDD（NASA POWER 要求）；非法格式/非法日期返回 400"""
    s = (s or "").strip()
    if not s:
        return default
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        try:
            datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, f"日期无效: {s}")
        return f"{s[:4]}{s[5:7]}{s[8:10]}"
    if len(s) == 8 and s.isdigit():
        try:
            datetime.strptime(s, "%Y%m%d")
        except ValueError:
            raise HTTPException(400, f"日期无效: {s}")
        return s
    raise HTTPException(400, f"日期格式无效: {s}（应为 YYYY-MM-DD 或 YYYYMMDD）")

@app.get("/weather/{lat}/{lon}")
def weather_check(lat: float, lon: float):
    """查询位置的 NASA POWER 气象数据可用性"""
    _check_coords(lat, lon)
    try:
        from pcse.input import NASAPowerWeatherDataProvider
        wdp = NASAPowerWeatherDataProvider(latitude=lat, longitude=lon)
        return {
            "latitude": lat,
            "longitude": lon,
            "elevation_m": round(wdp.elevation, 1),
            "data_range": f"{wdp.first_date} ~ {wdp.last_date}",
            "years_available": wdp.last_date.year - wdp.first_date.year,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/weather/current/{lat}/{lon}")
def weather_current(lat: float, lon: float):
    """实时气温与天气 (Open-Meteo, 免费免key)"""
    _check_coords(lat, lon)
    try:
        url = (f"https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}"
               f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m"
               f"&timezone=Asia%2FShanghai")
        data = http_get_json(url, timeout=20)
        cur = data.get("current", {})
        return {
            "location": {"lat": round(lat, 4), "lon": round(lon, 4)},
            "time": cur.get("time"),
            "temperature_c": cur.get("temperature_2m"),
            "apparent_temperature_c": cur.get("apparent_temperature"),
            "humidity_pct": cur.get("relative_humidity_2m"),
            "precipitation_mm": cur.get("precipitation"),
            "wind_speed_kmh": cur.get("wind_speed_10m"),
            "elevation_m": data.get("elevation"),
            "source": "Open-Meteo 实时",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"实时天气获取失败: {str(e)}")

@app.get("/weather/forecast/{lat}/{lon}")
def weather_forecast(lat: float, lon: float, days: int = 7):
    """未来天气预报 (Open-Meteo, 最多16天)"""
    _check_coords(lat, lon)
    try:
        days = max(1, min(days, 16))
        url = (f"https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}"
               f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max"
               f"&forecast_days={days}"
               f"&timezone=Asia%2FShanghai")
        data = http_get_json(url, timeout=20)
        daily = data.get("daily", {})
        result = {
            "location": {"lat": round(lat, 4), "lon": round(lon, 4)},
            "forecast_days": days,
            "source": "Open-Meteo 预报",
            "daily": [],
        }
        times = daily.get("time", [])
        for i, t in enumerate(times):
            result["daily"].append({
                "date": t,
                "max_temp_c": _daily_val(daily, "temperature_2m_max", i),
                "min_temp_c": _daily_val(daily, "temperature_2m_min", i),
                "precipitation_mm": _daily_val(daily, "precipitation_sum", i),
                "precip_probability_pct": _daily_val(daily, "precipitation_probability_max", i),
                "max_wind_kmh": _daily_val(daily, "wind_speed_10m_max", i),
            })
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"预报获取失败: {str(e)}")

@app.get("/weather/et0/{lat}/{lon}")
def weather_et0(lat: float, lon: float, days: int = 7):
    """蒸散量 ET0 (FAO Penman-Monteith, 灌溉决策核心)"""
    _check_coords(lat, lon)
    try:
        days = max(1, min(days, 16))
        url = (f"https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}"
               f"&daily=et0_fao_evapotranspiration,temperature_2m_max,temperature_2m_min,precipitation_sum"
               f"&forecast_days={days}"
               f"&timezone=Asia%2FShanghai")
        data = http_get_json(url, timeout=20)
        daily = data.get("daily", {})
        result = {
            "location": {"lat": round(lat, 4), "lon": round(lon, 4)},
            "source": "Open-Meteo FAO ET0",
            "unit": "mm/day",
            "daily": [],
            "summary": {},
        }
        times = daily.get("time", [])
        et0s = daily.get("et0_fao_evapotranspiration", [])
        for i, t in enumerate(times):
            et0 = et0s[i] if i < len(et0s) else None
            precip = _daily_val(daily, "precipitation_sum", i)
            result["daily"].append({
                "date": t,
                "et0_mm": et0,
                "max_temp_c": _daily_val(daily, "temperature_2m_max", i),
                "precipitation_mm": precip,
                "net_water_need_mm": round(et0 - (precip or 0), 1) if et0 is not None and precip is not None else None,
            })
        if et0s:
            valid = [e for e in et0s if e is not None]
            if valid:
                result["summary"] = {
                    "avg_et0_mm_per_day": round(sum(valid)/len(valid), 2),
                    "total_et0_mm": round(sum(valid), 1),
                }
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"蒸散量获取失败: {str(e)}")

@app.get("/weather/history/{lat}/{lon}/{start}/{end}")
def weather_history(lat: float, lon: float, start: str, end: str):
    """历史天气 (Open-Meteo ERA5 archive)"""
    _check_coords(lat, lon)
    try:
        # 格式统一为 YYYY-MM-DD
        def fmt_date(s):
            s = s.strip()
            if len(s) == 8 and s.isdigit():
                return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
            return s
        start_f, end_f = fmt_date(start), fmt_date(end)
        url = (f"https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={lat}&longitude={lon}"
               f"&start_date={start_f}&end_date={end_f}"
               f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
               f"&timezone=Asia%2FShanghai")
        data = http_get_json(url, timeout=25)
        daily = data.get("daily", {})
        result = {
            "location": {"lat": round(lat, 4), "lon": round(lon, 4)},
            "period": {"start": start, "end": end},
            "source": "Open-Meteo ERA5 历史",
            "daily": [],
        }
        times = daily.get("time", [])
        for i, t in enumerate(times):
            result["daily"].append({
                "date": t,
                "max_temp_c": _daily_val(daily, "temperature_2m_max", i),
                "min_temp_c": _daily_val(daily, "temperature_2m_min", i),
                "precipitation_mm": _daily_val(daily, "precipitation_sum", i),
            })
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"历史天气获取失败: {str(e)}")

@app.get("/weather/air/{lat}/{lon}")
def weather_air(lat: float, lon: float):
    """空气质量 + 紫外线 (Open-Meteo CAMS, 免费免key)"""
    _check_coords(lat, lon)
    try:
        url = (f"https://air-quality-api.open-meteo.com/v1/air-quality"
               f"?latitude={lat}&longitude={lon}"
               f"&current=pm10,pm2_5,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide,uv_index,us_aqi"
               f"&timezone=Asia%2FShanghai")
        data = http_get_json(url, timeout=20)
        cur = data.get("current", {})
        return {
            "location": {"lat": round(lat, 4), "lon": round(lon, 4)},
            "time": cur.get("time"),
            "source": "Open-Meteo CAMS",
            "pm10_ugm3": cur.get("pm10"),
            "pm2_5_ugm3": cur.get("pm2_5"),
            "no2_ugm3": cur.get("nitrogen_dioxide"),
            "ozone_ugm3": cur.get("ozone"),
            "so2_ugm3": cur.get("sulphur_dioxide"),
            "co_ugm3": cur.get("carbon_monoxide"),
            "uv_index": cur.get("uv_index"),
            "us_aqi": cur.get("us_aqi"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"空气质量获取失败: {str(e)}")

@app.get("/geocode/{name}")
def geocode(name: str):
    """地名转经纬度 (Open-Meteo 地理编码)"""
    try:
        import urllib.parse
        q = urllib.parse.quote(name)
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=5&language=zh&format=json"
        data = http_get_json(url, timeout=20)
        results = []
        for r in (data.get("results") or []):
            results.append({
                "name": r.get("name"),
                "latitude": r.get("latitude"),
                "longitude": r.get("longitude"),
                "admin1": r.get("admin1"),
                "country": r.get("country"),
                "population": r.get("population"),
            })
        return {"query": name, "source": "Open-Meteo 地理编码", "results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"地理编码失败: {str(e)}")

@app.get("/exchange-rate/{currency}")
def exchange_rate(currency: str = "CNY"):
    """实时汇率 (免费免key)"""
    try:
        url = "https://open.er-api.com/v6/latest/CNY"
        data = http_get_json(url, timeout=20)
        rates = data.get("rates", {})
        target = currency.upper()
        result = {
            "base": "CNY",
            "source": "ExchangeRate-API",
            "updated": data.get("time_last_update_utc"),
            "rates": {},
        }
        if target == "ALL":
            for k in ["USD", "EUR", "JPY", "GBP", "RUB", "KZT", "HKD", "AED"]:
                if k in rates:
                    result["rates"][k] = rates[k]
        elif target in rates:
            result["rates"][target] = rates[target]
        else:
            result["rates"] = {target: None}
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"汇率获取失败: {str(e)}")

@app.get("/gis/geocode/amap/{address}")
def gis_geocode_amap(address: str):
    """高德地理编码 (需 AMAP_KEY 环境变量)"""
    amap_key = os.environ.get("AMAP_KEY", "")
    if not amap_key:
        return {"error": "未配置高德地图 Key", "howto": "到 https://lbs.amap.com 注册免费key，设置环境变量 AMAP_KEY"}
    try:
        import urllib.parse
        q = urllib.parse.quote(address)
        url = f"https://restapi.amap.com/v3/geocode/geo?address={q}&key={amap_key}"
        data = http_get_json(url, timeout=20)
        return {
            "query": address,
            "source": "高德地图",
            "geocodes": data.get("geocodes", []),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"高德地理编码失败: {str(e)}")

@app.get("/soil/{lat}/{lon}")
def soil_check(lat: float, lon: float):
    """查询位置的土壤属性 (ISRIC SoilGrids)"""
    _check_coords(lat, lon)
    try:
        soildata, soil_info = soilgrids_to_soildata(lat, lon)
        return {
            "location": {"lat": lat, "lon": lon},
            "soil": soil_info,
            "wofost_params": soildata,
            "cached": bool(soil_info.get("cached")),
        }
    except Exception as e:
        return {
            "location": {"lat": lat, "lon": lon},
            "error": f"土壤数据获取失败: {str(e)}",
            "fallback": "将使用默认壤土参数",
        }

@app.get("/soil-moisture/{lat}/{lon}")
def soil_moisture(lat: float, lon: float, start: Optional[str] = None, end: Optional[str] = None):
    """查询位置的墒情时间序列 (NASA POWER)"""
    _check_coords(lat, lon)
    try:
        return fetch_soil_moisture(lat, lon, start, end)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"墒情数据获取失败: {str(e)}")

@app.get("/gis/elevation/{lat}/{lon}")
def gis_elevation(lat: float, lon: float):
    """查询位置高程"""
    _check_coords(lat, lon)
    return fetch_elevation(lat, lon)

@app.get("/gis/boundary/{code}")
def gis_boundary(code: str):
    """行政区划 GeoJSON (如 650000=新疆, 653100=喀什地区)"""
    return fetch_boundary(code)

# ========== 模拟结果缓存（同参数1小时内复用，避免重复计算）==========
_sim_cache = {}
SIM_CACHE_TTL = 3600
_sim_lock = threading.Lock()

@app.post("/simulate", response_model=SimResult)
def simulate(req: SimRequest):
    """运行 WOFOST 作物模拟（自动使用真实土壤参数，带缓存）"""
    import hashlib
    cache_key = hashlib.md5(
        json.dumps(req.model_dump(), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    with _sim_lock:  # 防并发同参数重复计算（thundering herd）
        cached = _sim_cache.get(cache_key)
        if cached and time.time() - cached["ts"] < SIM_CACHE_TTL:
            cached["data"].cached = True
            return cached["data"]

        result = _simulate_impl(req)
        result.cached = False
        _sim_cache[cache_key] = {"ts": time.time(), "data": result}
        if len(_sim_cache) > 200:
            for k in list(_sim_cache)[:50]:
                _sim_cache.pop(k, None)
    return result

def _simulate_impl(req: SimRequest):
    """实际模拟计算（原逻辑）"""
    t0 = time.time()
    tmp = None
    soil_info = None
    try:
        # 1. 天气
        from pcse.input import NASAPowerWeatherDataProvider, YAMLAgroManagementReader
        wdp = NASAPowerWeatherDataProvider(latitude=req.latitude, longitude=req.longitude)

        # 2. 真实土壤参数 (SoilGrids)，失败则用默认
        try:
            soildata, soil_info = soilgrids_to_soildata(req.latitude, req.longitude, timeout=55, retries=1)
        except Exception as e:
            print(f"[simulate] SoilGrids 获取失败，使用默认土壤: {e}")
            soil_info = {"soil_texture": "默认壤土(参考)", "used_default": True, "error": str(e)[:100]}
            soildata = None

        # 3. 作物参数
        params, crop_name = load_params(req.crop_no, wav=req.initial_soil_moisture, soildata=soildata)

        # 4. 农事管理
        agro_yaml = build_agro_yaml(req.planting_date, crop_name)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".agro", delete=False)
        tmp.write(agro_yaml)
        tmp.close()
        agro = YAMLAgroManagementReader(tmp.name)

        # 5. 模型
        from pcse.models import Wofost72_PP, Wofost72_WLP_CWB
        ModelClass = Wofost72_PP if req.production_mode == "pp" else Wofost72_WLP_CWB
        model = ModelClass(params, wdp, agro)
        model.run_till_terminate()

        # 6. 输出
        output = model.get_output()
        summary = model.get_summary_output()
        df = pd.DataFrame(output)
        ds = pd.DataFrame(summary) if summary else pd.DataFrame()

        summary_dict = {}
        if not ds.empty:
            for col in ds.columns:
                val = ds[col].values[0]
                if isinstance(val, (datetime, date)):
                    summary_dict[col] = str(val)
                elif val is None:
                    summary_dict[col] = None
                else:
                    summary_dict[col] = round(float(val), 2)

        key_vars = ["DVS", "LAI", "TAGP", "TWSO", "SM", "TRA"]
        if df.empty or "day" not in df.columns:
            # WOFOST 零输出（作物未出苗/生长季条件不成立）：返回空序列而非 KeyError
            timeseries = {v: [] for v in ["day"] + key_vars}
            return SimResult(
                duration_seconds=round(time.time() - t0, 3),
                summary=summary_dict,
                timeseries=timeseries,
                soil_info=soil_info,
                message=f"模拟未产生生长数据（{crop_name}：播种条件或生长季不成立），请检查播种日期与地区是否匹配。",
            )
        timeseries = {"day": [str(d) for d in df["day"].tolist()]}
        for var in key_vars:
            if var in df.columns:
                timeseries[var] = [
                    round(float(v), 2) if isinstance(v, (int, float)) else None
                    for v in df[var].tolist()
                ]

        elapsed = round(time.time() - t0, 3)
        tagp = summary_dict.get("TAGP", 0) or 0
        twso = summary_dict.get("TWSO", 0) or 0
        msg = f"模拟完成！{crop_name}"
        if tagp:
            msg += f"，总生物量 {tagp:.0f} kg/ha"
        if twso:
            msg += f"，产量 {twso:.0f} kg/ha"
        if soil_info and not soil_info.get("used_default"):
            msg += f"，土壤: {soil_info.get('soil_texture', '')}"

        if tmp:
            os.unlink(tmp.name)
        return SimResult(
            duration_seconds=elapsed,
            summary=summary_dict,
            timeseries=timeseries,
            soil_info=soil_info,
            message=msg,
        )

    except Exception as e:
        if tmp and os.path.exists(tmp.name):
            os.unlink(tmp.name)
        # 只返回用户可读消息，不泄漏服务端 traceback/内部路径
        raise HTTPException(500, detail=f"模拟失败: {str(e)[:200]}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
