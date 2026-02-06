import streamlit as st
import hashlib
import json
import os
import base64
import requests
import datetime
from io import BytesIO
from PIL import Image, ImageDraw

# --- 1. 页面配置 ---
st.set_page_config(page_title="Nano Banana Pro - Final Fix", layout="wide")

# --- 2. 基础环境 ---
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

USERS_FILE = "users.json"
VECTOR_ENGINE_BASE = "https://api.vectorengine.ai/v1"

# CSS: 强制白底
st.markdown("""
<style>
    .stApp { background-color: #f5f5f7; }
    div[data-testid="stImage"] { background-color: white; }
    iframe { background-color: white; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; background-color: #FF6600; color: white; }
</style>
""", unsafe_allow_html=True)

# ==========================================
#              GitHub 数据同步
# ==========================================
def get_github_config():
    if "github_token" in st.secrets and "repo_name" in st.secrets:
        return st.secrets["github_token"], st.secrets["repo_name"]
    return None, None

def load_users_from_github():
    token, repo = get_github_config()
    if not token or not repo:
        if os.path.exists(USERS_FILE):
            try: with open(USERS_FILE, "r", encoding="utf-8") as f: return json.load(f)
            except: return {}
        return {}
    url = f"https://api.github.com/repos/{repo}/contents/{USERS_FILE}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            content = base64.b64decode(resp.json()["content"]).decode("utf-8")
            return json.loads(content)
        return {}
    except: return {}

def save_users_to_github(users):
    try: with open(USERS_FILE, "w", encoding="utf-8") as f: json.dump(users, f, indent=4)
    except: pass
    token, repo = get_github_config()
    if not token or not repo: return
    url = f"https://api.github.com/repos/{repo}/contents/{USERS_FILE}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    json_str = json.dumps(users, indent=4)
    content_b64 = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
    sha = None
    try:
        get_resp = requests.get(url, headers=headers)
        if get_resp.status_code == 200: sha = get_resp.json()["sha"]
    except: pass
    data = {"message": "Update users", "content": content_b64}
    if sha: data["sha"] = sha
    try: requests.put(url, headers=headers, json=data)
    except: pass

# ==========================================
#              核心工具：Base64 转换
# ==========================================

def image_to_base64(image):
    """把图片转成字符串，解决白屏的核心函数"""
    buffered = BytesIO()
    # 强制保存为 PNG
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def resize_for_canvas(image, canvas_width):
    w, h = image.size
    ratio = canvas_width / w
    new_h = int(h * ratio)
    return image.resize((canvas_width, new_h), Image.Resampling.LANCZOS).convert("RGB"), new_h

def compress_img(image, max_size=1024):
    img = image.copy().convert("RGB")
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def get_coords(res, ow, oh, cw, ch):
    if res.json_data is None: return []
    return [(int(r["left"]/cw*ow), int(r["top"]/ch*oh), int((r["left"]+r["width"])/cw*ow), int((r["top"]+r["height"])/ch*oh)) for r in res.json_data.get("objects", [])]

def draw_boxes(img, coords, color):
    if not coords: return img
    i = img.copy()
    draw = ImageDraw.Draw(i)
    for b in coords: draw.rectangle(b, outline=color, width=5)
    return i

def call_api(key, model, prompt, map_b64, feat_b64, clean_b64, fmt):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    sys_prompt = "You are an expert image editor. IMAGE 1: Location Map (RED BOXES) - Where to edit. IMAGE 2: Source Feature (BLUE BOXES) - What to copy. IMAGE 3: Clean Canvas - Draw here. RULE: Apply features from Img2 to Img3 at Img1 locations. OUTPUT: Clean image, NO RED BOXES."
    try:
        if fmt == "chat":
            payload = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "【Map (RED)】"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{map_b64}"}},
                        {"type": "text", "text": "【Source (BLUE)】"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{feat_b64}"}},
                        {"type": "text", "text": "【Canvas】"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{clean_b64}"}},
                        {"type": "text", "text": f"\n\n{sys_prompt}\nUSER: {prompt}"}
                    ]
                }], "max_tokens": 4096
            }
            ep = f"{VECTOR_ENGINE_BASE}/chat/completions"
        else:
            payload = {"model": model, "prompt": prompt + " NO RED BOXES", "image": f"data:image/jpeg;base64,{clean_b64}", "control_image": f"data:image/jpeg;base64,{map_b64}", "size": "1024x1024"}
            ep = f"{VECTOR_ENGINE_BASE}/images/generations"

        resp = requests.post(ep, headers=headers, json=payload, timeout=240)
        if resp.status_code != 200: return None, f"HTTP {resp.status_code}", resp.text
        
        rj = resp.json()
        url = None
        if "data" in rj and rj["data"]:
            d = rj["data"][0]
            url = d.get("url") or (f"data:image/jpeg;base64,{d['b64_json']}" if "b64_json" in d else None)
        if not url and "choices" in rj:
            c = rj["choices"][0]["message"]["content"]
            m = re.search(r'\((https?://\S+|data:image/[^;]+;base64,[^\)]+)\)', c)
            if m: url = m.group(1)
        return url, None, resp.text
    except Exception as e: return None, str(e), None

# ==========================================
#              应用入口
# ==========================================

def init_auth_state():
    if "user_info" not in st.session_state: st.session_state.user_info = None
    if "auth_page" not in st.session_state: st.session_state.auth_page = "login"

def login_page():
    st.markdown("<h2 style='text-align: center;'>🔐 Nano Banana Pro (Final Fix)</h2>", unsafe_allow_html=True)
    users = load_users_from_github()
    if not users: st.warning("⚠️ 请注册管理员账号")

    tabs = st.tabs(["登录", "注册账号"])
    with tabs[0]:
        with st.form("login"):
            u = st.text_input("用户名")
            p = st.text_input("密码", type="password")
            if st.form_submit_button("登录"):
                if u not in users: st.error("用户不存在")
                elif users[u]["password"] != hash_password(p): st.error("密码错误")
                elif not users[u].get("approved", False): st.error("🚫 待审核")
                else:
                    st.session_state.user_info = {"username": u, "role": users[u].get("role", "user")}
                    st.success("成功"); st.rerun()
    with tabs[1]:
        with st.form("reg"):
            nu = st.text_input("用户名")
            np = st.text_input("密码", type="password")
            np2 = st.text_input("确认密码", type="password")
            if st.form_submit_button("注册"):
                if not nu or not np: st.error("不能为空")
                elif np != np2: st.error("密码不一致")
                elif nu in users: st.error("已存在")
                else:
                    is_first = (len(users) == 0)
                    users[nu] = {
                        "password": hash_password(np),
                        "role": "admin" if is_first else "user",
                        "approved": True if is_first else False,
                        "created_at": str(datetime.datetime.now())
                    }
                    save_users_to_github(users)
                    if is_first: st.success("管理员注册成功"); st.rerun()
                    else: st.info("申请已提交，等待审核")

def admin_panel():
    if st.session_state.user_info and st.session_state.user_info["role"] == "admin":
        with st.sidebar.expander("🛡️ 管理员后台", expanded=False):
            users = load_users_from_github()
            dirty = False
            for u, d in users.items():
                if u == st.session_state.user_info["username"]: continue
                c1, c2 = st.columns([3,2])
                c1.text(f"{u} {'✅' if d['approved'] else '⏳'}")
                if not d['approved']:
                    if c2.button("通过", key=f"a_{u}"): users[u]['approved']=True; dirty=True
                else:
                    if c2.button("冻结", key=f"b_{u}"): users[u]['approved']=False; dirty=True
            if dirty: save_users_to_github(users); st.rerun()

def main_app():
    with st.sidebar:
        st.write(f"👤 **{st.session_state.user_info['username']}**")
        if st.button("退出"): st.session_state.user_info=None; st.rerun()
        st.markdown("---")
        admin_panel()
        if "init_cfg" not in st.session_state:
            if "ve_key" in st.secrets:
                st.session_state.k = st.secrets["ve_key"]
                st.session_state.m = st.secrets.get("ve_model", "gemini-2.0-flash-exp")
                st.session_state.f = st.secrets.get("api_format", "chat")
            st.session_state.init_cfg = True
        st.session_state.k = st.text_input("API Key", value=st.session_state.get("k", ""), type="password")
        st.session_state.m = st.text_input("Model ID", value=st.session_state.get("m", ""))
        st.session_state.f = st.radio("Mode", ["chat", "image"], index=0 if st.session_state.get("f")=="chat" else 1)

    st.markdown("<h1 style='text-align: center; color: #FF6600;'>🍌 Nano Banana Pro · Final Fix</h1>", unsafe_allow_html=True)
    if not CANVAS_AVAILABLE: st.error("依赖未安装"); st.stop()

    c1, c2 = st.columns(2)
    CANVAS_WIDTH = 400
    
    with c1:
        f1 = st.file_uploader("图1 (场景)", type=["jpg","png"], key="u1")
        if f1:
            if "last_f1" not in st.session_state or st.session_state.last_f1 != f1.name:
                st.session_state.img1 = Image.open(f1).convert("RGB")
                st.session_state.last_f1 = f1.name
    with c2:
        f2 = st.file_uploader("图2 (商品)", type=["jpg","png"], key="u2")
        if f2:
             if "last_f2" not in st.session_state or st.session_state.last_f2 != f2.name:
                st.session_state.img2 = Image.open(f2).convert("RGB")
                st.session_state.last_f2 = f2.name

    if "img1" in st.session_state and "img2" in st.session_state:
        st.markdown("---")
        cc1, cc2 = st.columns(2)
        
        # 调整尺寸
        disp_img1, h_can1 = resize_for_canvas(st.session_state.img1, CANVAS_WIDTH)
        disp_img2, h_can2 = resize_for_canvas(st.session_state.img2, CANVAS_WIDTH)
        
        # 【关键】转成 Base64 字符串
        bg_url1 = image_to_base64(disp_img1)
        bg_url2 = image_to_base64(disp_img2)

        with cc1:
            st.write("👉 **框选位置 (红框)**")
            # 传 Base64 字符串给画板，Canvas 3.1.3 支持这个
            res1 = st_canvas(
                fill_color="rgba(255, 0, 0, 0.2)", 
                stroke_width=1, stroke_color="#FF0000", 
                background_color="#ffffff",
                background_image=bg_url1,
                height=h_can1, width=CANVAS_WIDTH, 
                drawing_mode="rect", key=f"c1_{st.session_state.last_f1}"
            )
            
        with cc2:
            st.write("👉 **框选特征 (蓝框)**")
            res2 = st_canvas(
                fill_color="rgba(0, 0, 255, 0.2)", 
                stroke_width=1, stroke_color="#0000FF", 
                background_color="#ffffff",
                background_image=bg_url2,
                height=h_can2, width=CANVAS_WIDTH, 
                drawing_mode="rect", key=f"c2_{st.session_state.last_f2}"
            )

        prompt = st.text_area("提示词", height=80, placeholder="例如：把图2的商品放入图1的红框位置")
        
        if st.button("🚀 开始生成", type="primary"):
            if not st.session_state.k: st.error("请配置 Key")
            elif not prompt: st.warning("请输入提示词")
            else:
                with st.spinner("正在合成中，请稍候..."):
                    boxes1 = get_coords(res1, st.session_state.img1.width, st.session_state.img1.height, CANVAS_WIDTH, h_can1)
                    boxes2 = get_coords(res2, st.session_state.img2.width, st.session_state.img2.height, CANVAS_WIDTH, h_can2)
                    
                    ib1 = compress_img(draw_boxes(st.session_state.img1, boxes1, "#FF0000") if boxes1 else st.session_state.img1)
                    ib2 = compress_img(draw_boxes(st.session_state.img2, boxes2, "#0000FF") if boxes2 else st.session_state.img2)
                    ic = compress_img(st.session_state.img1)
                    
                    url, err, raw = call_
