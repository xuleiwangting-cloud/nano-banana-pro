import streamlit as st
import hashlib
import json
import os
import time
import base64
import re
import requests
import datetime
from io import BytesIO
from PIL import Image, ImageDraw

# --- 1. 页面配置 (必须在最前) ---
st.set_page_config(page_title="Nano Banana Pro - V4.0 Secure", layout="wide")

# --- 2. 基础环境与依赖 ---
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

# 全局配置文件路径
CONFIG_FILE = "config.json"
USERS_FILE = "users.json"
VECTOR_ENGINE_BASE = "https://api.vectorengine.ai/v1"

# CSS 样式优化
st.markdown("""
<style>
    .stApp { background-color: #f5f5f7; }
    .log-container {
        max-height: 300px;
        overflow-y: auto;
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 10px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        white-space: pre-wrap;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        background-color: #FF6600;
        color: white;
    }
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
#              安全鉴权模块
# ==========================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4)

def init_auth_state():
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "login"

def login_page():
    st.markdown("<h2 style='text-align: center;'>🔐 Nano Banana Pro 安全登录</h2>", unsafe_allow_html=True)
    
    users = load_users()
    
    # 如果没有用户，提示注册管理员
    if not users:
        st.warning("⚠️ 系统暂无用户，请先注册管理员账号。")
        st.session_state.auth_page = "register"

    tabs = st.tabs(["登录", "注册账号"])
    
    with tabs[0]:
        with st.form("login_form"):
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            submit = st.form_submit_button("登录")
            
            if submit:
                if username not in users:
                    st.error("用户不存在")
                elif users[username]["password"] != hash_password(password):
                    st.error("密码错误")
                elif not users[username].get("approved", False):
                    st.error("🚫 账号待审核：请联系管理员开通权限")
                else:
                    st.session_state.user_info = {
                        "username": username,
                        "role": users[username].get("role", "user")
                    }
                    st.success("登录成功！")
                    st.rerun()

    with tabs[1]:
        with st.form("register_form"):
            new_user = st.text_input("设置用户名")
            new_pass = st.text_input("设置密码", type="password")
            new_pass2 = st.text_input("确认密码", type="password")
            reg_submit = st.form_submit_button("注册")
            
            if reg_submit:
                if not new_user or not new_pass:
                    st.error("用户名和密码不能为空")
                elif new_pass != new_pass2:
                    st.error("两次输入的密码不一致")
                elif new_user in users:
                    st.error("用户名已存在")
                else:
                    # 第一个注册的用户自动成为管理员且无需审核
                    is_first_user = (len(users) == 0)
                    role = "admin" if is_first_user else "user"
                    approved = True if is_first_user else False
                    
                    users[new_user] = {
                        "password": hash_password(new_pass),
                        "role": role,
                        "approved": approved,
                        "created_at": str(datetime.datetime.now())
                    }
                    save_users(users)
                    if approved:
                        st.success("🎉 管理员账号注册成功！请前往登录页登录。")
                    else:
                        st.info("✅ 注册申请已提交！请等待管理员审核通过。")

def admin_sidebar_panel():
    """管理员控制面板"""
    if st.session_state.user_info and st.session_state.user_info["role"] == "admin":
        with st.sidebar.expander("🛡️ 管理员后台", expanded=False):
            st.write("用户管理")
            users = load_users()
            dirty = False
            
            for u, data in users.items():
                # 不显示自己
                if u == st.session_state.user_info["username"]:
                    continue
                
                col1, col2 = st.columns([3, 2])
                col1.text(f"{u} ({'✅' if data['approved'] else '⏳'})")
                
                if not data['approved']:
                    if col2.button("通过", key=f"app_{u}"):
                        users[u]['approved'] = True
                        dirty = True
                else:
                    if col2.button("冻结", key=f"ban_{u}"):
                        users[u]['approved'] = False
                        dirty = True
            
            if dirty:
                save_users(users)
                st.success("状态已更新")
                time.sleep(1)
                st.rerun()

# ==========================================
#              核心功能模块 (V4.0)
# ==========================================

# --- 辅助函数 ---
def log_message(msg, type="info"):
    if "logs" not in st.session_state: st.session_state.logs = []
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] [{type.upper()}] {msg}")

def compress_image_for_api(image, max_size=1024, quality=90):
    img = image.copy()
    if img.mode != "RGB": img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_all_canvas_coords(canvas_result, orig_w, orig_h, canvas_w, canvas_h):
    if canvas_result.json_data is None: return []
    objects = canvas_result.json_data.get("objects", [])
    if not objects: return []
    coords_list = []
    for r in objects:
        sx = int(r["left"] / canvas_w * orig_w)
        sy = int(r["top"] / canvas_h * orig_h)
        sw = int(r["width"] / canvas_w * orig_w)
        sh = int(r["height"] / canvas_h * orig_h)
        coords_list.append((sx, sy, sx+sw, sy+sh))
    return coords_list

def draw_all_visual_boxes(image, coords_list, color):
    if not coords_list: return image
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    for box in coords_list:
        draw.rectangle(box, outline=color, width=8) 
    return img_copy

def call_image_generation(api_key, model_name, user_prompt, img_location_map_b64, img_source_feat_b64, img_clean_canvas_b64, api_format):
    log_message(f"🚀 发起请求 - 模型: {model_name}", "info")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    # 三图分离系统指令
    system_instruction = """
    【CRITICAL INSTRUCTION】
    You are an expert image editor. You will receive 3 images to perform a "Local Feature Transfer" task.
    IMAGE 1: "Location Map" (Contains RED BOXES). Function: ONLY tells you COORDINATES to edit. DO NOT copy red boxes.
    IMAGE 2: "Source Feature" (Contains BLUE BOXES). Function: Tells you WHAT visual features to copy.
    IMAGE 3: "Clean Canvas" (Original Image). Function: This is your drawing canvas.
    **RULE:** Apply features from Image 2 onto Image 3 at the locations specified by Image 1.
    **OUTPUT:** The final image must be clean like Image 3. NO RED BOXES allowed!
    """
    final_prompt = f"{system_instruction}\n\nUSER COMMAND: {user_prompt}"
    
    try:
        if api_format == "chat":
            payload = {
                "model": model_name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "【Image 1: Location Map (RED BOXES)】"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_location_map_b64}"}},
                        {"type": "text", "text": "\n\n【Image 2: Source Feature (BLUE BOXES)】"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_source_feat_b64}"}},
                        {"type": "text", "text": "\n\n【Image 3: Clean Canvas (Target)】"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_clean_canvas_b64}"}},
                        {"type": "text", "text": f"\n\n{final_prompt}"}
                    ]
                }],
                "max_tokens": 4096, "temperature": 0.55
            }
            endpoint = f"{VECTOR_ENGINE_BASE}/chat/completions"
        else:
            payload = {
                "model": model_name, "prompt": final_prompt + " DO NOT DRAW RED BOXES.",
                "image": f"data:image/jpeg;base64,{img_clean_canvas_b64}", 
                "control_image": f"data:image/jpeg;base64,{img_location_map_b64}",
                "size": "1024x1024", "n": 1
            }
            endpoint = f"{VECTOR_ENGINE_BASE}/images/generations"

        log_message("⏳ 数据发送中...", "info")
        response = requests.post(endpoint, headers=headers, json=payload, timeout=240)
        
        if response.status_code != 200:
            log_message(f"❌ API Error: {response.status_code}", "error")
            return None, f"HTTP {response.status_code}", response.text

        res_json = response.json()
        img_url = None
        if "data" in res_json and res_json["data"]:
            d = res_json["data"][0]
            if "url" in d: img_url = d["url"]
            elif "b64_json" in d: img_url = f"data:image/jpeg;base64,{d['b64_json']}"
            
        if not img_url and "choices" in res_json:
            content = res_json["choices"][0]["message"]["content"]
            md_match = re.search(r'!\[.*?\]\((https?://\S+|data:image/[^;]+;base64,[^\)]+)\)', content)
            if md_match: img_url = md_match.group(1)
            else:
                url_match = re.search(r'(https?://[^\s\)"\'<>]+)', content)
                if url_match: img_url = url_match.group(1)
        
        if img_url: return img_url, None, response.text
        return None, "解析失败", response.text
    except Exception as e:
        return None, f"程序异常: {str(e)}", None

# ==========================================
#              主应用逻辑
# ==========================================

def main_app():
    # --- 侧边栏 ---
    with st.sidebar:
        st.write(f"👤 当前用户: **{st.session_state.user_info['username']}**")
        if st.button("🚪 退出登录"):
            st.session_state.user_info = None
            st.rerun()
            
        st.markdown("---")
        # 管理员面板
        admin_sidebar_panel()
        
        st.title("⚙️ 工作室配置")
        
        # 加载配置
        if "init_config" not in st.session_state:
            if os.path.exists(CONFIG_FILE):
                try:
                    with open(CONFIG_FILE, "r", encoding='utf-8') as f:
                        config = json.load(f)
                        st.session_state.ve_key = config.get("ve_key", "")
                        st.session_state.ve_model = config.get("ve_model", "")
                        st.session_state.api_format = config.get("api_format", "chat")
                except: pass
            st.session_state.init_config = True

        st.session_state.ve_key = st.text_input("API 密钥 (Key)", value=st.session_state.get("ve_key", ""), type="password")
        st.session_state.ve_model = st.text_input("模型 ID", value=st.session_state.get("ve_model", ""), placeholder="gemini-2.0-flash-exp")
        api_fmt = st.radio("调用模式", ["chat", "image"], index=0 if st.session_state.get("api_format")=="chat" else 1)
        st.session_state.api_format = api_fmt
        
        if st.button("💾 保存配置"):
            with open(CONFIG_FILE, "w", encoding='utf-8') as f:
                json.dump({"ve_key": st.session_state.ve_key, "ve_model": st.session_state.ve_model, "api_format": api_fmt}, f)
            st.success("配置已保存")
        
        st.markdown("---")
        if st.button("🗑️ 清空日志"):
            st.session_state.logs = []
            st.rerun()
        if "logs" in st.session_state:
            st.markdown(f'<div class="log-container">{"<br>".join(st.session_state.logs[::-1])}</div>', unsafe_allow_html=True)

    # --- 主界面 ---
    st.markdown("<h1 style='text-align: center; color: #FF6600;'>🍌 Nano Banana Pro · 电商专用版</h1>", unsafe_allow_html=True)

    if not CANVAS_AVAILABLE:
        st.error("请安装依赖: pip install streamlit-drawable-canvas")
        st.stop()

    c1, c2 = st.columns(2)
    CANVAS_WIDTH = 400

    # 图1 上传与处理
    with c1:
        f1 = st.file_uploader("📂 图1", type=["jpg", "png"], key="u1")
        # 修复加载问题：强制存入 Session State
        if f1: 
            img1 = Image.open(f1).convert("RGB")
            st.session_state.cached_img1 = img1
        elif "cached_img1" in st.session_state and not f1:
             # 如果用户删除了文件，清除缓存
             del st.session_state.cached_img1

    # 图2 上传与处理
    with c2:
        f2 = st.file_uploader("📂 图2", type=["jpg", "png"], key="u2")
        if f2: 
            img2 = Image.open(f2).convert("RGB")
            st.session_state.cached_img2 = img2
        elif "cached_img2" in st.session_state and not f2:
             del st.session_state.cached_img2

    # 渲染画板区域
    if "cached_img1" in st.session_state and "cached_img2" in st.session_state:
        st.markdown("---")
        cc1, cc2 = st.columns(2)
        
        with cc1:
            st.markdown("**图1操作：框选 (红框)**")
            img1 = st.session_state.cached_img1
            w1, h1 = img1.size
            h_can1 = int(h1 * (CANVAS_WIDTH/w1))
            
            # 修复加载问题：Key 绑定 file_uploader 的 ID，确保切换图片时重绘
            key1 = f"can1_{f1.name if f1 else 'default'}"
            res1 = st_canvas(
                fill_color="rgba(255, 0, 0, 0.1)", stroke_width=2, stroke_color="#FF0000", 
                background_image=img1, height=h_can1, width=CANVAS_WIDTH, 
                drawing_mode="rect", key=key1
            )
            
        with cc2:
            st.markdown("**图2操作：框选 (蓝框)**")
            img2 = st.session_state.cached_img2
            w2, h2 = img2.size
            h_can2 = int(h2 * (CANVAS_WIDTH/w2))
            
            # 修复加载问题：动态 Key
            key2 = f"can2_{f2.name if f2 else 'default'}"
            res2 = st_canvas(
                fill_color="rgba(0, 0, 255, 0.1)", stroke_width=2, stroke_color="#0000FF", 
                background_image=img2, height=h_can2, width=CANVAS_WIDTH, 
                drawing_mode="rect", key=key2
            )

        st.markdown("---")
        prompt = st.text_area("💬 提示词", value="", placeholder="例如：把图2的商品放入图1的所有红框位置...", height=80)
        st.write("") 
        btn_start = st.button("🚀 开始执行", type="primary")

        if btn_start:
            if not st.session_state.ve_key or not st.session_state.ve_model:
                st.error("❌ 请检查配置")
            elif not prompt.strip():
                st.warning("⚠️ 请输入提示词")
            else:
                if "result_image" not in st.session_state: st.session_state.result_image = None
                status = st.status("正在处理...", expanded=True)
                
                boxes1 = get_all_canvas_coords(res1, w1, h1, CANVAS_WIDTH, h_can1)
                boxes2 = get_all_canvas_coords(res2, w2, h2, CANVAS_WIDTH, h_can2)
                
                status.write(f"✂️ 正在构建逻辑...")
                img_clean_canvas_b64 = compress_image_for_api(img1)
                img1_boxed = draw_all_visual_boxes(img1, boxes1, "#FF0000") if boxes1 else img1
                img_location_map_b64 = compress_image_for_api(img1_boxed)
                img2_boxed = draw_all_visual_boxes(img2, boxes2, "#0000FF") if boxes2 else img2
                img_source_feat_b64 = compress_image_for_api(img2_boxed)

                status.write(f"📡 发送请求 ({st.session_state.ve_model})...")
                img_url, err_msg, raw_resp = call_image_generation(
                    st.session_state.ve_key, st.session_state.ve_model, prompt,
                    img_location_map_b64, img_source_feat_b64, img_clean_canvas_b64,
                    st.session_state.api_format
                )
                
                if img_url:
                    st.session_state.result_image = img_url
                    status.update(label="✅ 执行成功!", state="complete")
                else:
                    status.update(label="❌ 失败", state="error")
                    st.error(f"❌ 错误: {err_msg}")
                    with st.expander("🔍 查看原因"):
                        st.code(raw_resp, language="json")

    # 结果展示区
    if "result_image" in st.session_state and st.session_state.result_image:
        st.markdown("---")
        col_show, col_dl = st.columns([3, 1])
        with col_show:
            st.image(st.session_state.result_image, caption="结果", use_column_width=True)
        with col_dl:
            st.success("✅ 图片已就绪")
            if st.session_state.result_image.startswith("data:image"):
                try:
                    header, base64_data = st.session_state.result_image.split(",", 1)
                    img_bytes = base64.b64decode(base64_data)
                    try:
                        pil_img = Image.open(BytesIO(img_bytes))
                        buf = BytesIO()
                        pil_img.save(buf, format="PNG")
                        final_bytes = buf.getvalue()
                        final_ext = "png"
                        final_mime = "image/png"
                    except:
                        final_bytes = img_bytes
                        final_ext = "png" if "png" in header.lower() else "jpg"
                        final_mime = "image/png" if "png" in header.lower() else "image/jpeg"
                    st.download_button(label=f"📥 下载图片 (.{final_ext})", data=final_bytes, 
                                     file_name=f"result_{int(datetime.datetime.now().timestamp())}.{final_ext}", mime=final_mime)
                except Exception as e: st.error(f"下载错误: {e}")
            else:
                st.link_button("📥 打开图片链接", st.session_state.result_image)

# ==========================================
#              程序入口
# ==========================================

init_auth_state()

if st.session_state.user_info is None:
    login_page()
else:
    main_app()
