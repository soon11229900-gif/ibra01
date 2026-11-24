// ============================================================================
//  WARZONE 3D  -  Android NativeActivity game, pure C++ / OpenGL ES 3.0
//  Third-person shooter (toggle to first-person), waves of enemies,
//  rifle + shotgun, virtual joystick, fire / jump / reload / view / weapon buttons.
// ============================================================================
#include <android_native_app_glue.h>
#include <android/log.h>
#include <jni.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>

#define LOG(...) __android_log_print(ANDROID_LOG_INFO, "Warzone", __VA_ARGS__)

// ------------------------------------------------------------------ math ----
struct V3 { float x, y, z; };
static V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static float len(V3 a) { return sqrtf(dot(a, a)); }
static V3 norm(V3 a) { float l = len(a); return l > 1e-6f ? a * (1.0f / l) : V3{0, 0, 0}; }
static V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static float frand(float a, float b) { return a + (b - a) * (rand() / (float)RAND_MAX); }
static float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }

struct M4 { float m[16]; };  // column-major
static M4 ident() { M4 r; memset(&r, 0, sizeof r); r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1; return r; }
static M4 mul(const M4& a, const M4& b) {
    M4 r;
    for (int c = 0; c < 4; c++)
        for (int rr = 0; rr < 4; rr++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a.m[k * 4 + rr] * b.m[c * 4 + k];
            r.m[c * 4 + rr] = s;
        }
    return r;
}
static M4 trans(V3 t) { M4 r = ident(); r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z; return r; }
static M4 scal(V3 s) { M4 r = ident(); r.m[0] = s.x; r.m[5] = s.y; r.m[10] = s.z; return r; }
static M4 rotY(float a) { M4 r = ident(); float c = cosf(a), s = sinf(a); r.m[0] = c; r.m[2] = -s; r.m[8] = s; r.m[10] = c; return r; }
static M4 rotX(float a) { M4 r = ident(); float c = cosf(a), s = sinf(a); r.m[5] = c; r.m[6] = s; r.m[9] = -s; r.m[10] = c; return r; }
static M4 perspective(float fovy, float asp, float n, float f) {
    M4 r; memset(&r, 0, sizeof r);
    float t = 1.0f / tanf(fovy * 0.5f);
    r.m[0] = t / asp; r.m[5] = t; r.m[10] = (f + n) / (n - f); r.m[11] = -1; r.m[14] = 2 * f * n / (n - f);
    return r;
}
static M4 lookAt(V3 e, V3 c, V3 up) {
    V3 f = norm(c - e), s = norm(cross(f, up)), u = cross(s, f);
    M4 r = ident();
    r.m[0] = s.x; r.m[4] = s.y; r.m[8] = s.z;
    r.m[1] = u.x; r.m[5] = u.y; r.m[9] = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -dot(s, e); r.m[13] = -dot(u, e); r.m[14] = dot(f, e);
    return r;
}

// ---------------------------------------------------------------- shaders ---
static const char* VS3 = R"(#version 300 es
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aNrm;
uniform mat4 uVP; uniform mat4 uM; uniform vec3 uScale;
out vec3 vN; out vec3 vW;
void main(){ vec4 w = uM * vec4(aPos,1.0); vW = w.xyz; vN = mat3(uM) * (aNrm / uScale); gl_Position = uVP * w; })";
static const char* FS3 = R"(#version 300 es
precision mediump float;
in vec3 vN; in vec3 vW; out vec4 o;
uniform vec3 uColor; uniform vec3 uCam; uniform vec3 uFog; uniform float uEmit; uniform float uGrid;
void main(){
  vec3 n = normalize(vN);
  float d = max(dot(n, normalize(vec3(0.4,1.0,0.3))), 0.0);
  float l = 0.38 + 0.62 * d;
  vec3 c = uColor * l + uColor * uEmit;
  if (uGrid > 0.5) { float t = mod(floor(vW.x/4.0) + floor(vW.z/4.0), 2.0); c *= 0.88 + 0.12 * t; }
  float f = clamp((length(vW - uCam) - 40.0) / 80.0, 0.0, 1.0);
  o = vec4(mix(c, uFog, f), 1.0);
})";
static const char* VS2 = R"(#version 300 es
layout(location=0) in vec2 aPos; layout(location=1) in vec4 aCol;
uniform vec2 uRes; out vec4 vC;
void main(){ vC = aCol; gl_Position = vec4(aPos.x/uRes.x*2.0-1.0, 1.0-aPos.y/uRes.y*2.0, 0.0, 1.0); })";
static const char* FS2 = R"(#version 300 es
precision mediump float;
in vec4 vC; out vec4 o;
void main(){ o = vC; })";

// ------------------------------------------------------------ GL / EGL state
static android_app* gApp = nullptr;
static EGLDisplay dpy = EGL_NO_DISPLAY;
static EGLSurface surf = EGL_NO_SURFACE;
static EGLContext ctx = EGL_NO_CONTEXT;
static int SW = 1, SH = 1;
static bool hasWindow = false, focused = false;
static GLuint p3 = 0, p2 = 0, vaoCube = 0, vboCube = 0, vaoHud = 0, vboHud = 0;
static GLint uVP, uM, uScale, uColor, uCam, uFog, uEmit, uGrid, uRes;

static GLuint compile(GLenum t, const char* s) {
    GLuint sh = glCreateShader(t);
    glShaderSource(sh, 1, &s, nullptr);
    glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) { char b[512]; glGetShaderInfoLog(sh, 512, nullptr, b); LOG("shader error: %s", b); }
    return sh;
}
static GLuint program(const char* vs, const char* fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(p);
    return p;
}
static void initGL() {
    p3 = program(VS3, FS3);
    uVP = glGetUniformLocation(p3, "uVP"); uM = glGetUniformLocation(p3, "uM");
    uScale = glGetUniformLocation(p3, "uScale"); uColor = glGetUniformLocation(p3, "uColor");
    uCam = glGetUniformLocation(p3, "uCam"); uFog = glGetUniformLocation(p3, "uFog");
    uEmit = glGetUniformLocation(p3, "uEmit"); uGrid = glGetUniformLocation(p3, "uGrid");
    p2 = program(VS2, FS2);
    uRes = glGetUniformLocation(p2, "uRes");

    // unit cube, 36 vertices, (pos, normal)
    float v[36 * 6]; int n = 0;
    for (int f = 0; f < 6; f++) {
        int axis = f / 2, a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
        float sg = (f % 2 == 0) ? 1.f : -1.f;
        const int cu[4] = {-1, 1, 1, -1}, cv[4] = {-1, -1, 1, 1};
        const int idx[6] = {0, 1, 2, 0, 2, 3};
        for (int i = 0; i < 6; i++) {
            float pos[3], nr[3] = {0, 0, 0};
            pos[axis] = 0.5f * sg; pos[a1] = 0.5f * cu[idx[i]]; pos[a2] = 0.5f * cv[idx[i]];
            nr[axis] = sg;
            for (int k = 0; k < 3; k++) v[n++] = pos[k];
            for (int k = 0; k < 3; k++) v[n++] = nr[k];
        }
    }
    glGenVertexArrays(1, &vaoCube); glBindVertexArray(vaoCube);
    glGenBuffers(1, &vboCube); glBindBuffer(GL_ARRAY_BUFFER, vboCube);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, (void*)12);

    glGenVertexArrays(1, &vaoHud); glBindVertexArray(vaoHud);
    glGenBuffers(1, &vboHud); glBindBuffer(GL_ARRAY_BUFFER, vboHud);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 24, (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 24, (void*)8);
    glBindVertexArray(0);
}

static void hideSystemUi() {
    if (!gApp) return;
    JNIEnv* env = nullptr;
    JavaVM* vm = gApp->activity->vm;
    if (vm->AttachCurrentThread(&env, nullptr) != JNI_OK || !env) return;
    jobject act = gApp->activity->clazz;
    jclass ac = env->GetObjectClass(act);
    jmethodID gw = env->GetMethodID(ac, "getWindow", "()Landroid/view/Window;");
    jobject win = env->CallObjectMethod(act, gw);
    jclass wc = env->GetObjectClass(win);
    jmethodID gd = env->GetMethodID(wc, "getDecorView", "()Landroid/view/View;");
    jobject dec = env->CallObjectMethod(win, gd);
    jclass vc = env->GetObjectClass(dec);
    jmethodID sv = env->GetMethodID(vc, "setSystemUiVisibility", "(I)V");
    env->CallVoidMethod(dec, sv, 0x1706);  // immersive sticky, hide nav + status
    if (env->ExceptionCheck()) env->ExceptionClear();
    vm->DetachCurrentThread();
}

static bool initEGL() {
    dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    eglInitialize(dpy, nullptr, nullptr);
    const EGLint ca[] = {EGL_RENDERABLE_TYPE, 0x40 /*ES3*/, EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
                         EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_NONE};
    EGLConfig cfg; EGLint n = 0;
    if (!eglChooseConfig(dpy, ca, &cfg, 1, &n) || n < 1) { LOG("no EGL config"); return false; }
    EGLint fmt; eglGetConfigAttrib(dpy, cfg, EGL_NATIVE_VISUAL_ID, &fmt);
    ANativeWindow_setBuffersGeometry(gApp->window, 0, 0, fmt);
    surf = eglCreateWindowSurface(dpy, cfg, gApp->window, nullptr);
    const EGLint xa[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, xa);
    if (!eglMakeCurrent(dpy, surf, surf, ctx)) { LOG("eglMakeCurrent failed"); return false; }
    eglSwapInterval(dpy, 1);
    initGL();
    return true;
}
static void termEGL() {
    if (dpy != EGL_NO_DISPLAY) {
        eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (ctx != EGL_NO_CONTEXT) eglDestroyContext(dpy, ctx);
        if (surf != EGL_NO_SURFACE) eglDestroySurface(dpy, surf);
        eglTerminate(dpy);
    }
    dpy = EGL_NO_DISPLAY; ctx = EGL_NO_CONTEXT; surf = EGL_NO_SURFACE;
}

// -------------------------------------------------------------- game data ---
static const float ARENA = 40.f;
struct Box { V3 c, h, col; };  // center, half extents, color
struct Wpn { const char* name; int mag; float rate, reload; int pellets; float dmg, spread, kick; bool autof; };
static const Wpn WPN[2] = {
    {"RIFLE", 30, 0.09f, 1.7f, 1, 22.f, 0.010f, 0.010f, true},
    {"SHOTGUN", 6, 0.85f, 2.2f, 8, 13.f, 0.060f, 0.060f, false}};

struct Enemy { V3 p; float yaw, hp, cd, phase, flash, strafe, strafeT, deadT; bool alive; };
struct Tracer { V3 a, b; float life; V3 col; };

static std::vector<Box> boxes;
static std::vector<Enemy> enemies;
static std::vector<Tracer> tracers;

static struct {
    V3 p; float vy, yaw, pitch, hp, cd, reload, flash, hurt, hitT, phase;
    int weapon, ammo[2]; bool ground;
} P;
static bool fps = false, dead = false, waitingWave = true;
static int wave = 0, score = 0, kills = 0;
static float waveTimer = 1.5f, bannerT = 0;
static V3 camPos = {0, 2, -5}, camDir = {0, 0, 1};
static const V3 MUZ = {0.2f, 1.25f, 1.15f};  // gun muzzle in character space

static V3 worldMuz(V3 pos, float yaw, V3 l) {
    return pos + V3{l.x * cosf(yaw) + l.z * sinf(yaw), l.y, -l.x * sinf(yaw) + l.z * cosf(yaw)};
}

static void addBox(V3 c, V3 h, V3 col) { boxes.push_back({c, h, col}); }
static void initWorld() {
    boxes.clear();
    V3 wall = {0.45f, 0.42f, 0.38f};
    addBox({0, 2, ARENA + 0.5f}, {ARENA + 1, 2, 0.5f}, wall);
    addBox({0, 2, -ARENA - 0.5f}, {ARENA + 1, 2, 0.5f}, wall);
    addBox({ARENA + 0.5f, 2, 0}, {0.5f, 2, ARENA + 1}, wall);
    addBox({-ARENA - 0.5f, 2, 0}, {0.5f, 2, ARENA + 1}, wall);
    V3 bld = {0.5f, 0.5f, 0.56f};
    addBox({-15, 2, -10}, {3, 2, 3.5f}, bld);
    addBox({18, 2, -14}, {4, 2, 3}, bld);
    addBox({-20, 2, 16}, {3.5f, 2, 3}, bld);
    addBox({12, 2, 20}, {3, 2, 4}, bld);
    srand(12345);
    for (int i = 0; i < 22; i++) {
        float x = frand(-34, 34), z = frand(-34, 34);
        if (fabsf(x) < 6 && fabsf(z) < 6) continue;
        float hx = frand(0.7f, 1.4f), hy = frand(0.4f, 0.6f), hz = frand(0.7f, 1.4f);
        addBox({x, hy, z}, {hx, hy, hz}, {0.55f, 0.38f, 0.2f});
    }
    srand((unsigned)time(nullptr));
}

static void pushOut(V3& p, float r, float footY) {
    for (auto& b : boxes) {
        if (footY >= b.c.y + b.h.y - 0.3f) continue;
        float dx = p.x - b.c.x, dz = p.z - b.c.z;
        float ox = b.h.x + r - fabsf(dx), oz = b.h.z + r - fabsf(dz);
        if (ox > 0 && oz > 0) {
            if (ox < oz) p.x += dx > 0 ? ox : -ox; else p.z += dz > 0 ? oz : -oz;
        }
    }
    p.x = clampf(p.x, -ARENA + 0.6f, ARENA - 0.6f);
    p.z = clampf(p.z, -ARENA + 0.6f, ARENA - 0.6f);
}
static float groundAt(V3 p, float foot) {
    float g = 0;
    for (auto& b : boxes) {
        float top = b.c.y + b.h.y;
        if (top > foot + 0.4f) continue;
        if (fabsf(p.x - b.c.x) < b.h.x + 0.2f && fabsf(p.z - b.c.z) < b.h.z + 0.2f) g = std::max(g, top);
    }
    return g;
}

// ------------------------------------------------------------ ray casting ---
static bool rayBox(V3 o, V3 d, const Box& b, float& tOut) {
    float oo[3] = {o.x, o.y, o.z}, dd[3] = {d.x, d.y, d.z};
    float mn[3] = {b.c.x - b.h.x, b.c.y - b.h.y, b.c.z - b.h.z};
    float mx[3] = {b.c.x + b.h.x, b.c.y + b.h.y, b.c.z + b.h.z};
    float t0 = 0, t1 = 1e9f;
    for (int i = 0; i < 3; i++) {
        if (fabsf(dd[i]) < 1e-8f) { if (oo[i] < mn[i] || oo[i] > mx[i]) return false; }
        else {
            float inv = 1.0f / dd[i], a = (mn[i] - oo[i]) * inv, c = (mx[i] - oo[i]) * inv;
            if (a > c) std::swap(a, c);
            t0 = std::max(t0, a); t1 = std::min(t1, c);
            if (t0 > t1) return false;
        }
    }
    tOut = t0; return true;
}
static bool raySphere(V3 o, V3 d, V3 c, float r, float& t) {
    V3 oc = o - c; float b = dot(oc, d), cc = dot(oc, oc) - r * r, disc = b * b - cc;
    if (disc < 0) return false;
    float s = sqrtf(disc); t = -b - s; if (t < 0) t = -b + s;
    return t >= 0;
}
struct Hit { float t; int e; bool head; };
static Hit traceRay(V3 o, V3 d, float maxT) {
    Hit h{maxT, -1, false};
    for (auto& b : boxes) { float t; if (rayBox(o, d, b, t) && t < h.t) h.t = t; }
    if (d.y < -1e-4f) { float t = -o.y / d.y; if (t > 0 && t < h.t) h.t = t; }
    for (size_t i = 0; i < enemies.size(); i++) {
        Enemy& e = enemies[i]; if (!e.alive) continue;
        float t;
        if (raySphere(o, d, e.p + V3{0, 1.75f, 0}, 0.3f, t) && t < h.t) { h.t = t; h.e = (int)i; h.head = true; }
        if (raySphere(o, d, e.p + V3{0, 1.1f, 0}, 0.5f, t) && t < h.t) { h.t = t; h.e = (int)i; h.head = false; }
        if (raySphere(o, d, e.p + V3{0, 0.4f, 0}, 0.4f, t) && t < h.t) { h.t = t; h.e = (int)i; h.head = false; }
    }
    return h;
}

// ------------------------------------------------------------------ input ---
enum { R_NONE, R_JOY, R_LOOK, R_FIRE, R_JUMP, R_RELOAD, R_VIEW, R_WEAPON };
struct Touch { bool on = false; float x = 0, y = 0, lx = 0, ly = 0; int role = R_NONE; };
static const int MAXT = 10;
static Touch T[MAXT];
static float joyOx = 0, joyOy = 0, joyX = 0, joyY = 0;
static bool jumpReq = false, reloadReq = false, switchReq = false, restartReq = false;

struct C4 { float r, g, b, a; };
struct Btn { float x, y, r; int role; const char* label; C4 col; };
static int makeBtns(Btn* b) {
    float s = SH / 480.0f;
    b[0] = {SW - 100 * s, SH - 110 * s, 55 * s, R_FIRE, "FIRE", {0.9f, 0.2f, 0.15f, 0.55f}};
    b[1] = {SW - 210 * s, SH - 60 * s, 36 * s, R_JUMP, "JUMP", {0.2f, 0.7f, 0.3f, 0.55f}};
    b[2] = {SW - 210 * s, SH - 170 * s, 34 * s, R_RELOAD, "R", {0.2f, 0.45f, 0.9f, 0.55f}};
    b[3] = {SW - 60 * s, 50 * s, 30 * s, R_VIEW, "VIEW", {0.8f, 0.7f, 0.2f, 0.55f}};
    b[4] = {SW - 145 * s, 50 * s, 30 * s, R_WEAPON, "GUN", {0.7f, 0.4f, 0.8f, 0.55f}};
    return 5;
}
static void resetGame();
static void touchDown(int id, float x, float y) {
    if (id < 0 || id >= MAXT) return;
    if (dead) { restartReq = true; return; }
    Touch& t = T[id]; t.on = true; t.x = t.lx = x; t.y = t.ly = y; t.role = R_NONE;
    Btn b[5]; int n = makeBtns(b);
    for (int i = 0; i < n; i++) {
        float rr = b[i].r * 1.2f;
        if ((x - b[i].x) * (x - b[i].x) + (y - b[i].y) * (y - b[i].y) <= rr * rr) {
            t.role = b[i].role;
            if (t.role == R_JUMP) jumpReq = true;
            if (t.role == R_RELOAD) reloadReq = true;
            if (t.role == R_VIEW) fps = !fps;
            if (t.role == R_WEAPON) switchReq = true;
            return;
        }
    }
    if (x < SW * 0.45f) { t.role = R_JOY; joyOx = x; joyOy = y; joyX = joyY = 0; }
    else t.role = R_LOOK;
}
static void touchMove(int id, float x, float y) {
    if (id < 0 || id >= MAXT || !T[id].on) return;
    Touch& t = T[id]; t.x = x; t.y = y;
    float s = SH / 480.0f;
    if (t.role == R_JOY) {
        float dx = (x - joyOx) / (70 * s), dy = (y - joyOy) / (70 * s);
        float l = sqrtf(dx * dx + dy * dy);
        if (l > 1) { dx /= l; dy /= l; }
        joyX = dx; joyY = dy;
    } else if (t.role == R_LOOK || t.role == R_FIRE) {
        const float K = 0.0035f;
        P.yaw -= (x - t.lx) * K;
        P.pitch = clampf(P.pitch - (y - t.ly) * K, -1.2f, 1.2f);
        t.lx = x; t.ly = y;
    }
}
static void touchUp(int id) {
    if (id < 0 || id >= MAXT) return;
    if (T[id].role == R_JOY) joyX = joyY = 0;
    T[id].on = false; T[id].role = R_NONE;
}
static int32_t onInput(android_app*, AInputEvent* ev) {
    if (AInputEvent_getType(ev) != AINPUT_EVENT_TYPE_MOTION) return 0;
    int32_t a = AMotionEvent_getAction(ev);
    int act = a & AMOTION_EVENT_ACTION_MASK;
    int idx = (a & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
    size_t n = AMotionEvent_getPointerCount(ev);
    if (act == AMOTION_EVENT_ACTION_DOWN || act == AMOTION_EVENT_ACTION_POINTER_DOWN) {
        touchDown(AMotionEvent_getPointerId(ev, idx), AMotionEvent_getX(ev, idx), AMotionEvent_getY(ev, idx));
    } else if (act == AMOTION_EVENT_ACTION_MOVE) {
        for (size_t i = 0; i < n; i++)
            touchMove(AMotionEvent_getPointerId(ev, i), AMotionEvent_getX(ev, i), AMotionEvent_getY(ev, i));
    } else if (act == AMOTION_EVENT_ACTION_UP || act == AMOTION_EVENT_ACTION_POINTER_UP) {
        touchUp(AMotionEvent_getPointerId(ev, idx));
    } else if (act == AMOTION_EVENT_ACTION_CANCEL) {
        for (int i = 0; i < MAXT; i++) touchUp(i);
    }
    return 1;
}

// ------------------------------------------------------------- game logic ---
static V3 eyePos() { return P.p + V3{0, 1.6f, 0}; }

static void resetGame() {
    P.p = {0, 0, 0}; P.vy = 0; P.yaw = 0; P.pitch = 0; P.hp = 100; P.cd = 0; P.reload = 0;
    P.flash = P.hurt = P.hitT = P.phase = 0; P.weapon = 0; P.ground = true;
    P.ammo[0] = WPN[0].mag; P.ammo[1] = WPN[1].mag;
    enemies.clear(); tracers.clear();
    dead = false; wave = 0; score = 0; kills = 0; waitingWave = true; waveTimer = 1.5f; bannerT = 0;
    joyX = joyY = 0;
}
static void spawnWave() {
    int n = std::min(1 + wave * 2, 14);
    for (int i = 0; i < n; i++) {
        float a = frand(0, 6.2832f), r = frand(20, 32);
        Enemy e{};
        e.p = P.p + V3{sinf(a) * r, 0, cosf(a) * r};
        e.p.x = clampf(e.p.x, -ARENA + 2, ARENA - 2); e.p.z = clampf(e.p.z, -ARENA + 2, ARENA - 2);
        pushOut(e.p, 0.5f, 0);
        e.hp = 100; e.cd = frand(1.0f, 2.5f); e.alive = true; e.strafe = 1; e.strafeT = frand(1, 3);
        enemies.push_back(e);
    }
}
static void playerFire() {
    const Wpn& w = WPN[P.weapon];
    P.cd = w.rate; P.ammo[P.weapon]--; P.flash = 0.06f;
    P.pitch = clampf(P.pitch + w.kick * frand(0.6f, 1.2f), -1.2f, 1.2f);
    P.yaw += frand(-0.5f, 0.5f) * w.kick;

    V3 eye = eyePos();
    Hit ah = traceRay(camPos, camDir, 100.0f);
    V3 target = camPos + camDir * ah.t;
    V3 base = camDir;
    if (!fps && len(target - eye) > 1.0f) base = norm(target - eye);

    V3 right = {-cosf(P.yaw), 0, sinf(P.yaw)};
    V3 camUp = cross(right, camDir);
    V3 muz = fps ? (camPos + right * 0.22f + camUp * -0.2f + camDir * 0.85f) : worldMuz(P.p, P.yaw, MUZ);

    for (int i = 0; i < w.pellets; i++) {
        V3 d = norm(base + V3{frand(-1, 1), frand(-1, 1), frand(-1, 1)} * w.spread);
        Hit h = traceRay(eye, d, 120.0f);
        if (h.e >= 0) {
            Enemy& e = enemies[h.e];
            e.hp -= w.dmg * (h.head ? 2.5f : 1.0f); e.flash = 0.1f; P.hitT = 0.15f;
            if (e.hp <= 0 && e.alive) { e.alive = false; e.deadT = 0; kills++; score += 100 + (h.head ? 50 : 0); }
        }
        if (i < 3 || w.pellets == 1) tracers.push_back({muz, eye + d * h.t, 0.06f, {1.0f, 0.9f, 0.4f}});
    }
}
static void enemyFire(Enemy& e) {
    V3 muz = worldMuz(e.p, e.yaw, MUZ), tgt = P.p + V3{0, 1.2f, 0};
    V3 d = tgt - muz; float L = len(d); if (L < 0.1f) return; d = d * (1.0f / L);
    for (auto& b : boxes) { float t; if (rayBox(muz, d, b, t) && t < L - 0.5f) return; }  // blocked
    bool hit = frand(0, 1) < clampf(0.55f - L * 0.012f, 0.15f, 0.5f);
    V3 end = tgt;
    if (!hit) end = tgt + V3{frand(-1.5f, 1.5f), frand(-0.5f, 1.0f), frand(-1.5f, 1.5f)};
    tracers.push_back({muz, end, 0.08f, {1.0f, 0.35f, 0.1f}});
    if (hit) { P.hp -= 7; P.hurt = 0.4f; }
}

static void update(float dt) {
    if (dead) {
        if (restartReq) { restartReq = false; resetGame(); for (int i = 0; i < MAXT; i++) T[i] = Touch(); }
        return;
    }
    restartReq = false;
    const Wpn& w = WPN[P.weapon];
    P.cd = std::max(0.f, P.cd - dt); P.flash -= dt; P.hurt -= dt; P.hitT -= dt; bannerT -= dt;

    if (switchReq) { switchReq = false; P.weapon ^= 1; P.reload = 0; P.cd = 0.3f; }
    if (reloadReq) { reloadReq = false; if (P.reload <= 0 && P.ammo[P.weapon] < w.mag) P.reload = w.reload; }
    if (P.reload > 0) { P.reload -= dt; if (P.reload <= 0) P.ammo[P.weapon] = WPN[P.weapon].mag; }

    // movement
    float jl = sqrtf(joyX * joyX + joyY * joyY);
    V3 f = {sinf(P.yaw), 0, cosf(P.yaw)}, r = {-cosf(P.yaw), 0, sinf(P.yaw)};
    V3 mv = f * (-joyY) + r * joyX;
    float sp = jl > 0.9f ? 7.0f : 5.0f;
    P.p = P.p + mv * (sp * dt);
    pushOut(P.p, 0.4f, P.p.y);
    P.phase += jl * sp * dt * 1.6f;
    if (jumpReq) { jumpReq = false; if (P.ground) { P.vy = 8.0f; P.ground = false; } }
    P.p.y += P.vy * dt; P.vy -= 22.0f * dt;
    float g = groundAt(P.p, P.p.y);
    if (P.p.y <= g) { P.p.y = g; P.vy = 0; P.ground = true; } else P.ground = false;

    // shooting
    static bool prevFire = false;
    bool fireHeld = false;
    for (int i = 0; i < MAXT; i++) if (T[i].on && T[i].role == R_FIRE) fireHeld = true;
    bool want = fireHeld && (w.autof || !prevFire);
    prevFire = fireHeld;
    if (want && P.cd <= 0 && P.reload <= 0) {
        if (P.ammo[P.weapon] > 0) playerFire();
        else P.reload = w.reload;
    }

    // camera
    V3 eye = eyePos();
    camDir = {cosf(P.pitch) * sinf(P.yaw), sinf(P.pitch), cosf(P.pitch) * cosf(P.yaw)};
    if (fps) camPos = eye;
    else {
        V3 pivot = eye + r * 0.55f + V3{0, 0.15f, 0};
        camPos = pivot - camDir * 3.6f;
        camPos.y = std::max(camPos.y, 0.3f);
        camPos.x = clampf(camPos.x, -ARENA + 0.3f, ARENA - 0.3f);
        camPos.z = clampf(camPos.z, -ARENA + 0.3f, ARENA - 0.3f);
    }

    // waves
    int alive = 0; for (auto& e : enemies) if (e.alive) alive++;
    if (alive == 0) {
        if (!waitingWave) { waitingWave = true; waveTimer = 3.0f; if (wave > 0) P.hp = std::min(100.f, P.hp + 30.f); }
        waveTimer -= dt;
        if (waveTimer <= 0) { wave++; spawnWave(); bannerT = 2.5f; waitingWave = false; }
    }

    // enemies
    float espeed = 2.8f + std::min(wave, 10) * 0.15f;
    for (auto& e : enemies) {
        if (!e.alive) { e.deadT += dt; continue; }
        e.flash -= dt;
        V3 d = P.p - e.p; d.y = 0;
        float dist = len(d); V3 dn = norm(d);
        e.yaw = atan2f(d.x, d.z);
        e.strafeT -= dt; if (e.strafeT <= 0) { e.strafe = (rand() % 2) ? 1.f : -1.f; e.strafeT = frand(1, 3); }
        V3 rt = {-dn.z, 0, dn.x}, m = {0, 0, 0};
        if (dist > 14) m = dn;
        else if (dist > 7) m = dn * 0.4f + rt * (e.strafe * 0.6f);
        else if (dist < 4) m = dn * -0.6f + rt * (e.strafe * 0.6f);
        else m = rt * (e.strafe * 0.8f);
        e.p = e.p + m * (espeed * dt);
        for (auto& o : enemies) {
            if (&o == &e || !o.alive) continue;
            V3 dd = e.p - o.p; dd.y = 0; float l = len(dd);
            if (l < 0.9f && l > 0.001f) e.p = e.p + dd * ((0.9f - l) / l * 0.5f);
        }
        pushOut(e.p, 0.45f, 0);
        e.phase += len(m) * espeed * dt * 1.6f;
        e.cd -= dt;
        if (dist < 35 && e.cd <= 0) { enemyFire(e); e.cd = frand(0.7f, 1.5f); }
    }
    enemies.erase(std::remove_if(enemies.begin(), enemies.end(), [](const Enemy& e) { return !e.alive && e.deadT > 3.0f; }), enemies.end());

    for (auto& t : tracers) t.life -= dt;
    tracers.erase(std::remove_if(tracers.begin(), tracers.end(), [](const Tracer& t) { return t.life <= 0; }), tracers.end());

    if (P.hp <= 0) { P.hp = 0; dead = true; for (int i = 0; i < MAXT; i++) T[i] = Touch(); joyX = joyY = 0; }
}

// -------------------------------------------------------------- rendering ---
static void cube(const M4& model, V3 s, V3 c, float emit = 0, float grid = 0) {
    glUniformMatrix4fv(uM, 1, GL_FALSE, model.m);
    glUniform3f(uScale, s.x, s.y, s.z); glUniform3f(uColor, c.x, c.y, c.z);
    glUniform1f(uEmit, emit); glUniform1f(uGrid, grid);
    glDrawArrays(GL_TRIANGLES, 0, 36);
}
static void boxAt(V3 c, V3 s, V3 col, float emit = 0, float grid = 0) { cube(mul(trans(c), scal(s)), s, col, emit, grid); }
static void part(const M4& base, V3 off, V3 size, V3 col, float emit = 0) { cube(mul(base, mul(trans(off), scal(size))), size, col, emit); }
static void limb(const M4& base, V3 hip, float ang, V3 size, V3 col) {
    M4 m = mul(base, mul(trans(hip), mul(rotX(ang), trans({0, -size.y * 0.5f, 0}))));
    cube(mul(m, scal(size)), size, col);
}
static void drawHuman(const M4& base, V3 skin, V3 cloth, V3 pants, float phase, bool moving, float flash) {
    float sw = moving ? sinf(phase) * 0.7f : 0;
    if (flash > 0) { cloth = {1, 1, 1}; pants = {1, 1, 1}; }
    limb(base, {-0.15f, 0.8f, 0}, sw, {0.24f, 0.8f, 0.26f}, pants);
    limb(base, {0.15f, 0.8f, 0}, -sw, {0.24f, 0.8f, 0.26f}, pants);
    part(base, {0, 1.15f, 0}, {0.62f, 0.72f, 0.36f}, cloth);
    part(base, {0, 1.72f, 0}, {0.3f, 0.32f, 0.3f}, skin);
    part(base, {0, 1.9f, 0}, {0.36f, 0.12f, 0.36f}, {0.2f, 0.25f, 0.2f});
    part(base, {0.0f, 1.22f, 0.4f}, {0.15f, 0.15f, 0.5f}, cloth);
    part(base, {0.3f, 1.25f, 0.35f}, {0.15f, 0.15f, 0.6f}, cloth);
    part(base, {0.2f, 1.25f, 0.7f}, {0.08f, 0.12f, 0.9f}, {0.1f, 0.1f, 0.1f});
}
static void drawTracer(const Tracer& t) {
    V3 d = t.b - t.a; float L = len(d); if (L < 0.01f) return;
    float yaw = atan2f(d.x, d.z), pit = -asinf(clampf(d.y / L, -1, 1));
    M4 m = mul(trans((t.a + t.b) * 0.5f), mul(rotY(yaw), rotX(pit)));
    V3 s = {0.04f, 0.04f, L};
    cube(mul(m, scal(s)), s, t.col, 1.0f);
}

// ---- HUD batch (pixel space) ----
static std::vector<float> hv;
static void vtx(float x, float y, const C4& c) { hv.push_back(x); hv.push_back(y); hv.push_back(c.r); hv.push_back(c.g); hv.push_back(c.b); hv.push_back(c.a); }
static void rect(float x, float y, float w, float h, C4 c) {
    vtx(x, y, c); vtx(x + w, y, c); vtx(x + w, y + h, c);
    vtx(x, y, c); vtx(x + w, y + h, c); vtx(x, y + h, c);
}
static void circle(float cx, float cy, float r, C4 c) {
    const int N = 28;
    for (int i = 0; i < N; i++) {
        float a0 = i * 6.2832f / N, a1 = (i + 1) * 6.2832f / N;
        vtx(cx, cy, c); vtx(cx + cosf(a0) * r, cy + sinf(a0) * r, c); vtx(cx + cosf(a1) * r, cy + sinf(a1) * r, c);
    }
}
struct Glyph { char c; const char* r; };
static const Glyph FONT[] = {
    {'0', "111101101101111"}, {'1', "010110010010111"}, {'2', "110001010100111"}, {'3', "110001010001110"},
    {'4', "101101111001001"}, {'5', "111100110001110"}, {'6', "011100111101111"}, {'7', "111001010010010"},
    {'8', "111101111101111"}, {'9', "111101111001110"}, {'A', "010101111101101"}, {'C', "011100100100011"},
    {'D', "110101101101110"}, {'E', "111100110100111"}, {'F', "111100110100100"}, {'G', "011100101101011"},
    {'H', "101101111101101"}, {'I', "111010010010111"}, {'J', "001001001101010"}, {'K', "101101110101101"},
    {'L', "100100100100111"}, {'M', "101111111101101"}, {'N', "110101101101101"}, {'O', "010101101101010"},
    {'P', "110101110100100"}, {'R', "110101110101101"}, {'S', "011100010001110"}, {'T', "111010010010010"},
    {'U', "101101101101111"}, {'V', "101101101101010"}, {'W', "101101111111101"}, {'X', "101101010101101"},
    {'/', "001001010100100"}, {':', "000010000010000"}};
static void text(const char* s, float x, float y, float cell, C4 col, int align) {
    int n = (int)strlen(s);
    float w = n * 4 * cell - cell;
    float px = align == 1 ? x - w / 2 : (align == 2 ? x - w : x);
    for (int i = 0; i < n; i++, px += 4 * cell) {
        for (const Glyph& g : FONT)
            if (g.c == s[i]) {
                for (int k = 0; k < 15; k++) if (g.r[k] == '1') rect(px + (k % 3) * cell, y + (k / 3) * cell, cell, cell, col);
                break;
            }
    }
}
static void flushHud() {
    if (hv.empty()) return;
    glBindVertexArray(vaoHud); glBindBuffer(GL_ARRAY_BUFFER, vboHud);
    glBufferData(GL_ARRAY_BUFFER, hv.size() * sizeof(float), hv.data(), GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(hv.size() / 6));
    hv.clear();
}
static void drawHud() {
    float s = SH / 480.0f, cx = SW / 2.0f, cy = SH / 2.0f;
    char buf[64];
    if (P.hurt > 0) rect(0, 0, SW, SH, {0.8f, 0, 0, clampf(P.hurt, 0, 0.4f) * 0.9f});

    // crosshair
    C4 cc = P.hitT > 0 ? C4{1, 0.2f, 0.2f, 1} : C4{1, 1, 1, 0.85f};
    float g = 6 * s, l = 10 * s, th = 2 * s;
    rect(cx - g - l, cy - th / 2, l, th, cc); rect(cx + g, cy - th / 2, l, th, cc);
    rect(cx - th / 2, cy - g - l, th, l, cc); rect(cx - th / 2, cy + g, th, l, cc);

    // health bar
    rect(20 * s, 16 * s, 220 * s, 18 * s, {0, 0, 0, 0.5f});
    float hp = clampf(P.hp / 100.f, 0, 1);
    rect(22 * s, 18 * s, 216 * s * hp, 14 * s, {1.0f - hp, 0.2f + 0.7f * hp, 0.15f, 0.9f});
    snprintf(buf, sizeof buf, "HP %d", (int)P.hp);
    text(buf, 28 * s, 20 * s, 2.0f * s, {1, 1, 1, 1}, 0);

    // wave / score
    snprintf(buf, sizeof buf, "WAVE %d", wave);
    text(buf, cx, 14 * s, 3.0f * s, {1, 1, 1, 0.95f}, 1);
    snprintf(buf, sizeof buf, "SCORE %d  KILLS %d", score, kills);
    text(buf, cx, 38 * s, 2.0f * s, {1, 0.9f, 0.5f, 0.95f}, 1);

    // ammo
    const Wpn& w = WPN[P.weapon];
    text(w.name, SW - 20 * s, 100 * s, 2.4f * s, {1, 1, 1, 0.9f}, 2);
    snprintf(buf, sizeof buf, "%d/%d", P.ammo[P.weapon], w.mag);
    text(buf, SW - 20 * s, 122 * s, 4.0f * s, P.ammo[P.weapon] == 0 ? C4{1, 0.3f, 0.3f, 1} : C4{1, 1, 1, 1}, 2);
    if (P.reload > 0) {
        float pr = 1.0f - P.reload / w.reload;
        rect(SW - 160 * s, 150 * s, 140 * s, 8 * s, {0, 0, 0, 0.5f});
        rect(SW - 160 * s, 150 * s, 140 * s * pr, 8 * s, {0.3f, 0.6f, 1, 0.9f});
        text("RELOAD", SW - 20 * s, 164 * s, 2.0f * s, {0.6f, 0.8f, 1, 1}, 2);
    }

    // joystick
    bool joyOn = false;
    for (int i = 0; i < MAXT; i++) if (T[i].on && T[i].role == R_JOY) joyOn = true;
    float bx = joyOn ? joyOx : 130 * s, by = joyOn ? joyOy : SH - 120 * s;
    circle(bx, by, 70 * s, {1, 1, 1, 0.14f});
    circle(bx + joyX * 70 * s, by + joyY * 70 * s, 32 * s, {1, 1, 1, joyOn ? 0.5f : 0.3f});

    // buttons
    Btn b[5]; int n = makeBtns(b);
    for (int i = 0; i < n; i++) {
        bool down = false;
        for (int k = 0; k < MAXT; k++) if (T[k].on && T[k].role == b[i].role) down = true;
        C4 c = b[i].col; if (down) { c.a = 0.9f; }
        circle(b[i].x, b[i].y, b[i].r * 1.08f, {1, 1, 1, 0.25f});
        circle(b[i].x, b[i].y, b[i].r, c);
        float cell = (strlen(b[i].label) > 1 ? 2.4f : 5.0f) * s;
        text(b[i].label, b[i].x, b[i].y - 2.5f * cell, cell, {1, 1, 1, 1}, 1);
    }
    if (fps) text("1P", 20 * s, 44 * s, 2.0f * s, {1, 1, 1, 0.6f}, 0);
    else text("3P", 20 * s, 44 * s, 2.0f * s, {1, 1, 1, 0.6f}, 0);

    if (bannerT > 0 && !dead) {
        snprintf(buf, sizeof buf, "WAVE %d", wave);
        text(buf, cx, SH * 0.22f, 9.0f * s, {1, 0.85f, 0.3f, clampf(bannerT, 0, 1)}, 1);
    }
    if (dead) {
        rect(0, 0, SW, SH, {0, 0, 0, 0.6f});
        text("GAME OVER", cx, SH * 0.22f, 10.0f * s, {1, 0.3f, 0.2f, 1}, 1);
        snprintf(buf, sizeof buf, "SCORE %d", score);
        text(buf, cx, SH * 0.5f, 5.0f * s, {1, 1, 1, 1}, 1);
        text("TAP TO RESTART", cx, SH * 0.72f, 4.0f * s, {1, 0.9f, 0.5f, 1}, 1);
    }
}

static void render() {
    eglQuerySurface(dpy, surf, EGL_WIDTH, &SW); eglQuerySurface(dpy, surf, EGL_HEIGHT, &SH);
    glViewport(0, 0, SW, SH);
    const V3 sky = {0.55f, 0.7f, 0.9f};
    glClearColor(sky.x, sky.y, sky.z, 1); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST); glDisable(GL_CULL_FACE); glDisable(GL_BLEND);

    glUseProgram(p3); glBindVertexArray(vaoCube);
    M4 vp = mul(perspective(1.05f, (float)SW / SH, 0.1f, 200.f), lookAt(camPos, camPos + camDir, {0, 1, 0}));
    glUniformMatrix4fv(uVP, 1, GL_FALSE, vp.m);
    glUniform3f(uCam, camPos.x, camPos.y, camPos.z); glUniform3f(uFog, sky.x, sky.y, sky.z);

    boxAt({0, -0.5f, 0}, {200, 1, 200}, {0.35f, 0.46f, 0.28f}, 0, 1);
    for (auto& b : boxes) boxAt(b.c, b.h * 2.0f, b.col);

    for (auto& e : enemies) {
        float fall = e.alive ? 0 : std::min(1.0f, e.deadT * 3.0f) * 1.5f;
        M4 base = mul(trans(e.p), mul(rotY(e.yaw), rotX(-fall)));
        drawHuman(base, {0.9f, 0.7f, 0.55f}, {0.6f, 0.2f, 0.15f}, {0.25f, 0.22f, 0.2f}, e.phase, e.alive, e.flash);
    }

    V3 right = {-cosf(P.yaw), 0, sinf(P.yaw)};
    if (!fps) {
        M4 base = mul(trans(P.p), rotY(P.yaw));
        drawHuman(base, {0.9f, 0.72f, 0.58f}, {0.15f, 0.3f, 0.6f}, {0.2f, 0.2f, 0.25f}, P.phase, (joyX != 0 || joyY != 0), 0);
        if (P.flash > 0) boxAt(worldMuz(P.p, P.yaw, MUZ), {0.2f, 0.2f, 0.25f}, {1, 0.9f, 0.4f}, 1.5f);
    } else {
        V3 camUp = cross(right, camDir);
        float kick = P.flash > 0 ? 0.06f : 0;
        V3 gp = camPos + right * 0.22f + camUp * -0.2f + camDir * (0.55f - kick);
        M4 m = mul(trans(gp), mul(rotY(P.yaw), rotX(-P.pitch)));
        V3 gs = {0.08f, 0.1f, 0.55f};
        cube(mul(m, scal(gs)), gs, {0.12f, 0.12f, 0.12f});
        if (P.flash > 0) boxAt(gp + camDir * 0.4f, {0.14f, 0.14f, 0.14f}, {1, 0.9f, 0.4f}, 1.5f);
    }
    for (auto& t : tracers) drawTracer(t);

    // HUD
    glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(p2); glUniform2f(uRes, (float)SW, (float)SH);
    drawHud(); flushHud();
}

// -------------------------------------------------------------- app glue ----
static void onCmd(android_app* app, int32_t cmd) {
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            if (app->window) { hasWindow = initEGL(); focused = true; hideSystemUi(); }
            break;
        case APP_CMD_TERM_WINDOW: hasWindow = false; termEGL(); break;
        case APP_CMD_GAINED_FOCUS: focused = true; hideSystemUi(); break;
        case APP_CMD_LOST_FOCUS: focused = false; for (int i = 0; i < MAXT; i++) touchUp(i); break;
        default: break;
    }
}
static double nowSec() { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

void android_main(android_app* app) {
    gApp = app; app->onAppCmd = onCmd; app->onInputEvent = onInput;
    srand((unsigned)time(nullptr));
    initWorld(); resetGame();
    double last = nowSec();
    while (true) {
        int ev; android_poll_source* src;
        int timeout = (hasWindow && focused) ? 0 : -1;
        while (ALooper_pollOnce(timeout, nullptr, &ev, (void**)&src) >= 0) {
            if (src) src->process(app, src);
            if (app->destroyRequested) { termEGL(); return; }
            timeout = (hasWindow && focused) ? 0 : -1;
        }
        if (app->destroyRequested) { termEGL(); return; }
        if (hasWindow && focused && dpy != EGL_NO_DISPLAY) {
            double t = nowSec(); float dt = (float)(t - last); last = t;
            if (dt > 0.05f) dt = 0.05f;
            update(dt); render();
            eglSwapBuffers(dpy, surf);
        } else last = nowSec();
    }
}
