import argparse
import colorsys
import json
import math
import os
import random
import sys
from collections import Counter

import cv2
import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

CLASSES = ["plate", "bowl", "cup", "mug"]
STAGES = ["blank", "formed", "dried", "glazed", "fired", "defect"]

L = 6.2                     
BELT_HALF_WID = 0.45
SPAWN_X = -5.6                
PRESS0, PRESS1 = -4.3, -3.3  
DRY0, DRY1 = -2.9, -0.9       
GL0, GL1 = -0.1, 0.9       
K0, K1 = 1.5, 3.5           
INSP_C = 4.5                
PUSH_X = 5.25                 
PRESS_C = (PRESS0 + PRESS1) / 2
GL_C = (GL0 + GL1) / 2
CRACK_X = (K0 + K1) / 2     

STATION_X = {1: SPAWN_X, 2: PRESS_C, 3: (DRY0 + DRY1) / 2, 4: GL_C,
             5: (K0 + K1) / 2, 6: INSP_C, 7: PUSH_X}
STATION_NAMES = {1: "Бункер глини", 2: "Прес-формування", 3: "Сушка",
                 4: "Глазурування", 5: "Піч випалу", 6: "Контроль якості",
                 7: "Сортування/пакування"}

BELT_SPEED = 0.5
SIM_DT = 1.0 / 30.0
STRIPE_STEP = 0.25
DEFECT_PROB = 0.12
MORPH_LEVELS = 10

PROFILES = {
    "plate": [(0, 0), (0.09, 0), (0.095, 0.004), (0.19, 0.030), (0.195, 0.035),
              (0.19, 0.037), (0.09, 0.013), (0, 0.013)],
    "bowl": [(0, 0), (0.05, 0), (0.055, 0.008), (0.11, 0.040), (0.125, 0.085),
             (0.122, 0.090), (0.118, 0.087), (0.105, 0.048), (0.05, 0.018), (0, 0.015)],
    "cup": [(0, 0), (0.04, 0), (0.045, 0.006), (0.07, 0.07), (0.07, 0.075),
            (0.066, 0.075), (0.062, 0.07), (0.040, 0.014), (0, 0.012)],
    "mug": [(0, 0), (0.05, 0), (0.052, 0.004), (0.052, 0.10), (0.048, 0.10),
            (0.048, 0.008), (0, 0.008)],
}
HANDLES = {"cup": (0.064, 0.045, 0.022, 0.007), "mug": (0.052, 0.055, 0.030, 0.008)}

CRACKS = {
    "plate": ([(0.19, 0.037), (0.14, 0.025), (0.09, 0.013), (0.04, 0.013)], (0.0, 0.0015)),
    "bowl": ([(0.118, 0.087), (0.105, 0.048), (0.08, 0.033), (0.05, 0.018)], (-0.0015, 0.0015)),
    "cup": ([(0.07, 0.074), (0.07, 0.07), (0.057, 0.04), (0.047, 0.012)], (0.0015, 0.0)),
    "mug": ([(0.052, 0.098), (0.052, 0.07), (0.052, 0.04), (0.052, 0.015)], (0.0015, 0.0)),
}


def smooth(a, b, x):
    """Гладкий перехід smoothstep: 0 при x<=a, 1 при x>=b."""
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    return tuple(a[i] * (1 - t) + b[i] * t for i in range(3))


def item_look(x, glaze):
    """Колір/матеріал виробу як функція положення на лінії (стан процесу).
    Повертає (колір, specular, shininess, emission)."""
    wet = (0.42, 0.31, 0.24)           
    bisque = (0.86, 0.76, 0.63)            
    c = mix(wet, bisque, smooth(DRY0, DRY1, x))
    chalk = mix(glaze, (0.93, 0.93, 0.90), 0.5)   
    c = mix(c, chalk, smooth(GL0, GL1, x))
    kp = smooth(K0, K1, x)           
    c = mix(c, glaze, kp)             
    spec = 0.04 + 0.66 * kp
    shin = 8 + 85 * kp              
    hot = kp * (1 - smooth(K1, K1 + 0.8, x))     
    emis = (0.8 * hot, 0.25 * hot, 0.03 * hot)
    return c, spec, shin, emis


def random_ceramic_color(rng):
    if rng.random() < 0.35:
        v = rng.uniform(0.85, 1.0)
        return (v, v, v * rng.uniform(0.9, 1.0))
    return colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.3, 0.9), rng.uniform(0.35, 0.95))


def resample(poly, n):
    poly = np.asarray(poly, float)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]
    s = np.linspace(0, d[-1], n)
    return np.stack([np.interp(s, d, poly[:, 0]), np.interp(s, d, poly[:, 1])], 1)


def blank_profile(kind):
    """Глиняна заготовка (шайба) - початкова форма перед пресуванням."""
    pr = np.array(PROFILES[kind])
    rb = 0.65 * pr[:, 0].max() + 0.02
    hb = 0.55 * pr[:, 1].max() + 0.015
    return [(0, 0), (rb * 0.92, 0), (rb, hb * 0.3), (rb, hb * 0.7), (rb * 0.92, hb), (0, hb)]


def lathe_strips(profile, seg=48):
    th = np.linspace(0, 2 * np.pi, seg + 1)
    c, s = np.cos(th), np.sin(th)
    strips = []
    for (ra, ya), (rb, yb) in zip(profile[:-1], profile[1:]):
        dr, dy = rb - ra, yb - ya
        length = math.hypot(dr, dy)
        if length < 1e-9:
            continue
        nr, ny = dy / length, -dr / length
        v = np.zeros((seg + 1, 2, 3))
        n = np.zeros((seg + 1, 2, 3))
        for k, (r, y) in enumerate(((ra, ya), (rb, yb))):
            v[:, k, 0], v[:, k, 1], v[:, k, 2] = r * c, y, r * s
            n[:, k, 0], n[:, k, 1], n[:, k, 2] = nr * c, ny, nr * s
        strips.append((v, n))
    return strips


def handle_strips(x0, yc, big_r, small_r, nu=20, nv=10):
    u = np.linspace(-np.pi / 2, np.pi / 2, nu + 1)
    vs = np.linspace(0, 2 * np.pi, nv + 1)

    def pt(v):
        rad = big_r + small_r * np.cos(v)
        p = np.stack([x0 + rad * np.cos(u), yc + rad * np.sin(u),
                      np.full_like(u, small_r * np.sin(v))], axis=1)
        nrm = np.stack([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u),
                        np.full_like(u, np.sin(v))], axis=1)
        return p, nrm

    strips = []
    for j in range(nv):
        v = np.zeros((nu + 1, 2, 3))
        n = np.zeros((nu + 1, 2, 3))
        v[:, 0], n[:, 0] = pt(vs[j])
        v[:, 1], n[:, 1] = pt(vs[j + 1])
        strips.append((v, n))
    return strips


class Mesh:
    def __init__(self, strips):
        self.strips = strips
        self.points = np.concatenate([v.reshape(-1, 3) for v, _ in strips], axis=0)
        self.list_id = glGenLists(1)
        glNewList(self.list_id, GL_COMPILE)
        for v, n in strips:
            glBegin(GL_QUAD_STRIP)
            for k in range(v.shape[0]):
                for j in (0, 1):
                    glNormal3f(*n[k, j])
                    glVertex3f(*v[k, j])
            glEnd()
        glEndList()

    def draw(self):
        glCallList(self.list_id)


def build_mesh(kind, level):
    """Геометрія виробу на рівні пресування level (0 = заготовка, MAX = готова форма)."""
    final = np.array(PROFILES[kind], float)
    t = level / MORPH_LEVELS
    if level >= MORPH_LEVELS:
        prof = final
    else:
        a = resample(blank_profile(kind), 64)
        b = resample(final, 64)
        prof = (1 - t) * a + t * b
    strips = lathe_strips(prof)
    hs = smooth(0.8, 1.0, t)
    if kind in HANDLES and hs > 0.05:
        x0, yc, big_r, small_r = HANDLES[kind]
        strips += handle_strips(x0, yc, big_r * hs, small_r * hs)
    return Mesh(strips)


class Item:
    def __init__(self, rng, x, z):
        self.kind = rng.choice(CLASSES)
        self.glaze = random_ceramic_color(rng)
        self.x, self.y, self.z = x, 0.35, z        
        self.vy = 0.0
        self.angle = rng.uniform(0, 360)
        self.scale = rng.uniform(0.85, 1.15)
        self.defect = rng.random() < DEFECT_PROB
        self.crack_theta = rng.uniform(0, 2 * math.pi)
        self.crack_phase = rng.uniform(0, 6.28)
        self.state = "belt"                           

    def level(self):
        return int(round(smooth(PRESS0, PRESS1, self.x) * MORPH_LEVELS))

    def stage(self):
        x = self.x
        if x < (PRESS0 + PRESS1) / 2:
            return 0
        if x < (DRY0 + DRY1) / 2:
            return 1
        if x < (GL0 + GL1) / 2:
            return 2
        if x < (K0 + K1) / 2:
            return 3
        return 5 if self.defect else 4

    def world_points(self, mesh):
        a = math.radians(self.angle)
        c, s = math.cos(a), math.sin(a)
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return (mesh.points * self.scale) @ rot.T + np.array([self.x, self.y, self.z])


class Line:
    def __init__(self, rng):
        self.rng = rng
        self.items = []
        self.belt_offset = 0.0
        self.gap = rng.uniform(0.7, 1.1)
        self.speed = BELT_SPEED
        self.spawned = self.packed = self.rejected = 0

    def step(self, dt):
        d = self.speed * dt
        alive = []
        for it in self.items:
            if it.state == "belt":
                it.x += d
                if it.y > 0 or it.vy != 0:
                    it.vy -= 9.8 * dt
                    it.y += it.vy * dt
                    if it.y <= 0:
                        it.y, it.vy = 0.0, 0.0
                if it.defect and it.x >= PUSH_X:
                    it.state, it.x = "rejected", PUSH_X
                elif it.x > L:
                    it.state = "falling"
            elif it.state == "rejected":
                it.z += 0.9 * dt
                if it.z > BELT_HALF_WID + 0.12:
                    it.vy -= 9.8 * dt
                    it.y += it.vy * dt
            else:
                it.x += d
                it.vy -= 9.8 * dt
                it.y += it.vy * dt
            if it.y < -0.32:
                if it.state == "rejected":
                    self.rejected += 1
                else:
                    self.packed += 1
                continue
            alive.append(it)
        self.items = alive
        self.belt_offset = (self.belt_offset + d) % STRIPE_STEP
        belt = [it.x for it in self.items if it.state == "belt"]
        if not belt or min(belt) > SPAWN_X + self.gap:
            self.items.append(Item(self.rng, SPAWN_X, self.rng.uniform(-0.12, 0.12)))
            self.gap = self.rng.uniform(0.7, 1.1)
            self.spawned += 1


class View:
    def __init__(self):
        self.yaw, self.pitch, self.dist = 90.0, 30.0, 10.0
        self.target = np.array([0.0, 0.0, 0.0])
        self.fov = 45.0
        self.light = np.array([1.5, 3.5, 2.0])
        self.light_intensity = 1.0
        self.belt_shade = 0.18
        self.clear = (0.55, 0.58, 0.62)

    def eye(self):
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        return self.target + self.dist * np.array(
            [math.cos(p) * math.cos(y), math.sin(p), math.cos(p) * math.sin(y)])


class Renderer:
    def __init__(self):
        self.cache = {}
        self.quadric = None
        self.cube = None
        self.fx = random.Random(1)

    def init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glEnable(GL_MULTISAMPLE)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.quadric = gluNewQuadric()
        self.cube = glGenLists(1)
        glNewList(self.cube, GL_COMPILE)
        faces = [((0, 0, 1), [(-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)]),
                 ((0, 0, -1), [(.5, -.5, -.5), (-.5, -.5, -.5), (-.5, .5, -.5), (.5, .5, -.5)]),
                 ((1, 0, 0), [(.5, -.5, .5), (.5, -.5, -.5), (.5, .5, -.5), (.5, .5, .5)]),
                 ((-1, 0, 0), [(-.5, -.5, -.5), (-.5, -.5, .5), (-.5, .5, .5), (-.5, .5, -.5)]),
                 ((0, 1, 0), [(-.5, .5, .5), (.5, .5, .5), (.5, .5, -.5), (-.5, .5, -.5)]),
                 ((0, -1, 0), [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, -.5, .5), (-.5, -.5, .5)])]
        glBegin(GL_QUADS)
        for nrm, vs in faces:
            glNormal3f(*nrm)
            for v in vs:
                glVertex3f(*v)
        glEnd()
        glEndList()

    def get_mesh(self, kind, level):
        key = (kind, level)
        if key not in self.cache:
            self.cache[key] = build_mesh(kind, level)
        return self.cache[key]

    @staticmethod
    def material(color, spec=0.1, shin=20.0):
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (*color, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (spec, spec, spec, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shin)

    def box(self, c, size, color, spec=0.1, shin=20.0):
        self.material(color, spec, shin)
        glPushMatrix()
        glTranslatef(*c)
        glScalef(*size)
        glCallList(self.cube)
        glPopMatrix()

    def ghost(self, c, size, rgba):
        """Напівпрозорий об'єм (скло/кожух) - малюється після непрозорих."""
        glDisable(GL_LIGHTING)
        glDepthMask(GL_FALSE)
        glColor4f(*rgba)
        glPushMatrix()
        glTranslatef(*c)
        glScalef(*size)
        glCallList(self.cube)
        glPopMatrix()
        glDepthMask(GL_TRUE)
        glEnable(GL_LIGHTING)

    def portal(self, x, h, color, wz=0.55):
        for sz in (-1, 1):
            self.box((x, h / 2, sz * wz), (0.06, h, 0.06), color, 0.3, 30)
        self.box((x, h, 0), (0.07, 0.06, 2 * wz + 0.06), color, 0.3, 30)

    def vertical_cyl(self, x, y, z, radius, length, color):
        self.material(color, 0.6, 60)
        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(90, 1, 0, 0)
        gluCylinder(self.quadric, radius, radius, length, 16, 1)
        glPopMatrix()

    def open_crate(self, cx, cy, cz, sx, sy, sz, color):
        t = 0.02
        self.box((cx, cy - sy / 2, cz), (sx, t, sz), color)
        self.box((cx, cy, cz - sz / 2), (sx, sy, t), color)
        self.box((cx, cy, cz + sz / 2), (sx, sy, t), color)
        self.box((cx - sx / 2, cy, cz), (t, sy, sz), color)
        self.box((cx + sx / 2, cy, cz), (t, sy, sz), color)

    def draw_static(self, view, line):
        b = view.belt_shade
        steel, frame = (0.7, 0.72, 0.75), (0.25, 0.27, 0.3)
        self.box((0, -0.83, 0), (22, 0.02, 12), (0.35, 0.36, 0.38))
        self.box((0, -0.025, 0), (2 * L, 0.05, 2 * BELT_HALF_WID), (b, b, b * 1.05))
        self.material((b + 0.14,) * 3)
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        x = -L + line.belt_offset
        while x < L - 0.03:
            glVertex3f(x, 0.0015, -BELT_HALF_WID)
            glVertex3f(x + 0.03, 0.0015, -BELT_HALF_WID)
            glVertex3f(x + 0.03, 0.0015, BELT_HALF_WID)
            glVertex3f(x, 0.0015, BELT_HALF_WID)
            x += STRIPE_STEP
        glEnd()
        for sz in (-1, 1):
            z = sz * (BELT_HALF_WID + 0.025)
            a0, a1 = -L, PUSH_X - 0.25
            b0, b1 = PUSH_X + 0.25, L
            self.box(((a0 + a1) / 2, 0.03, z), (a1 - a0, 0.06, 0.05), steel, 0.8, 80)
            self.box(((b0 + b1) / 2, 0.03, z), (b1 - b0, 0.06, 0.05), steel, 0.8, 80)
        self.material(steel, 0.8, 80)
        for sx in (-1, 1):
            glPushMatrix()
            glTranslatef(sx * L, -0.05, -BELT_HALF_WID)
            gluCylinder(self.quadric, 0.05, 0.05, 2 * BELT_HALF_WID, 24, 1)
            glPopMatrix()
        for px in np.arange(-6.0, 6.1, 2.0):
            for sz in (-1, 1):
                self.box((px, -0.43, sz * 0.42), (0.06, 0.78, 0.06), frame, 0.3, 30)
        for sz in (-1, 1):
            self.box((0, -0.1, sz * 0.42), (2 * L, 0.05, 0.05), frame, 0.3, 30)

        orange = (0.9, 0.5, 0.1)
        self.box((SPAWN_X, 1.0, 0), (0.6, 0.35, 0.6), orange, 0.3, 30)
        self.box((SPAWN_X, 0.7, 0), (0.18, 0.3, 0.18), orange, 0.3, 30)
        for sx in (-1, 1):
            for sz in (-1, 1):
                self.box((SPAWN_X + sx * 0.27, 0.45, sz * 0.27), (0.04, 0.9, 0.04), frame)

        self.portal(PRESS_C, 0.78, (0.35, 0.4, 0.55))
        self.box((PRESS_C, 0.78, 0), (0.07, 0.06, 1.16), (0.35, 0.4, 0.55))
        depth = 0.0
        for it in line.items:
            if PRESS0 < it.x < PRESS1:
                depth = max(depth, math.sin(math.pi * smooth(PRESS0, PRESS1, it.x)))
        head_bottom = 0.45 - 0.35 * depth
        self.vertical_cyl(PRESS_C, 0.75, 0, 0.03, 0.75 - head_bottom - 0.03, steel)
        self.box((PRESS_C, head_bottom, 0), (0.36, 0.06, 0.36), (0.5, 0.52, 0.56), 0.6, 60)

        for x in (DRY0 - 0.2, DRY1 + 0.2, (DRY0 + DRY1) / 2):
            self.portal(x, 0.58, (0.3, 0.45, 0.75))
        for x in np.linspace(DRY0, DRY1, 4):
            self.box((x, 0.6, 0), (0.12, 0.04, 0.4), (0.95, 0.45, 0.1))

        self.portal(GL_C, 0.7, (0.2, 0.55, 0.3))
        for z in (-0.15, 0.0, 0.15):
            self.material(steel, 0.8, 80)
            glPushMatrix()
            glTranslatef(GL_C, 0.66, z)
            glRotatef(90, 1, 0, 0)
            gluCylinder(self.quadric, 0.03, 0.01, 0.1, 12, 1)
            glPopMatrix()

        for x in (K0 - 0.2, K1 + 0.2):
            self.portal(x, 0.6, (0.35, 0.12, 0.1))

        self.portal(INSP_C, 0.65, (0.85, 0.85, 0.9))
        self.box((INSP_C, 0.55, 0), (0.12, 0.1, 0.12), (0.1, 0.1, 0.12), 0.8, 80)
        self.box((INSP_C, 0.5, 0), (0.06, 0.03, 0.06), (0.02, 0.02, 0.05), 1.0, 100)
        bad = any(it.defect and abs(it.x - INSP_C) < 0.3 and it.state == "belt"
                  for it in line.items)
        glDisable(GL_LIGHTING)
        glColor3f(*((1, 0.1, 0.1) if bad else (0.1, 0.9, 0.2)))
        glPushMatrix()
        glTranslatef(INSP_C, 0.63, 0.35)
        gluSphere(self.quadric, 0.03, 12, 8)
        glPopMatrix()
        glEnable(GL_LIGHTING)

        pz = -0.62
        for it in line.items:
            if it.state == "rejected":
                pz = max(pz, it.z - 0.22)
        self.box((PUSH_X, 0.06, (pz - 0.9) / 2), (0.05, 0.05, abs(pz + 0.9)), steel, 0.8, 80)
        self.box((PUSH_X, 0.06, pz), (0.14, 0.12, 0.05), (0.9, 0.2, 0.15), 0.4, 40)
        self.open_crate(PUSH_X, -0.35, 1.05, 0.55, 0.5, 0.55, (0.8, 0.15, 0.15))
        self.open_crate(L + 0.5, -0.4, 0.0, 0.7, 0.5, 0.95, (0.65, 0.5, 0.3))

    def draw_items(self, line):
        for it in line.items:
            col, spec, shin, emis = item_look(it.x, it.glaze)
            mesh = self.get_mesh(it.kind, it.level())
            glPushMatrix()
            glTranslatef(it.x, it.y, it.z)
            glRotatef(it.angle, 0, 1, 0)
            glScalef(it.scale, it.scale, it.scale)
            self.material(col, spec, shin)
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, (*emis, 1.0))
            mesh.draw()
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, (0, 0, 0, 1.0))
            if it.defect and it.x > CRACK_X:
                self.draw_crack(it)
            glPopMatrix()

    @staticmethod
    def draw_crack(it):
        path, off = CRACKS[it.kind]
        pts = np.array(path)
        n = 13
        idx = np.linspace(0, len(pts) - 1, n)
        r = np.interp(idx, range(len(pts)), pts[:, 0]) + off[0]
        y = np.interp(idx, range(len(pts)), pts[:, 1]) + off[1]
        th = it.crack_theta + 0.12 * np.sin(np.arange(n) * 1.9 + it.crack_phase)
        glDisable(GL_LIGHTING)
        glColor3f(0.04, 0.03, 0.03)
        glLineWidth(3.0)
        glBegin(GL_LINE_STRIP)
        for k in range(n):
            glVertex3f(r[k] * math.cos(th[k]), y[k], r[k] * math.sin(th[k]))
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_shadows(self, line, light):
        Lh = np.array([*light, 1.0])
        S = Lh[1] * np.eye(4) - np.outer(Lh, np.array([0.0, 1.0, 0.0, 0.0]))
        glDisable(GL_LIGHTING)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-1.0, -1.0)
        glDepthMask(GL_FALSE)
        glColor4f(0, 0, 0, 0.35)
        for it in line.items:
            if it.x > L or abs(it.z) > BELT_HALF_WID - 0.02:
                continue
            glPushMatrix()
            glTranslatef(0, 0.002, 0)
            glMultMatrixf(S.T.astype(np.float32).flatten())
            glTranslatef(it.x, it.y, it.z)
            glRotatef(it.angle, 0, 1, 0)
            glScalef(it.scale, it.scale, it.scale)
            self.get_mesh(it.kind, it.level()).draw()
            glPopMatrix()
        glDepthMask(GL_TRUE)
        glDisable(GL_POLYGON_OFFSET_FILL)
        glEnable(GL_LIGHTING)

    def draw_translucent(self, line):
        self.ghost(((DRY0 + DRY1) / 2, 0.29, 0), (DRY1 - DRY0 + 0.4, 0.58, 1.1),
                   (0.6, 0.8, 1.0, 0.13))
        kc, kl = (K0 + K1) / 2, K1 - K0 + 0.4
        self.ghost((kc, 0.3, 0), (kl, 0.6, 1.1), (0.9, 0.25, 0.1, 0.22))
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 0.5, 0.1)
        for z in (-0.3, 0.0, 0.3):
            glPushMatrix()
            glTranslatef(K0 - 0.15, 0.48, z)
            glRotatef(90, 0, 1, 0)
            gluCylinder(self.quadric, 0.015, 0.015, kl - 0.1, 10, 1)
            glPopMatrix()
        glEnable(GL_LIGHTING)
        self.ghost((INSP_C, 0.3, 0), (0.7, 0.6, 1.1), (0.7, 1.0, 0.8, 0.06))
        if any(GL0 < it.x < GL1 and it.state == "belt" for it in line.items):
            glDisable(GL_LIGHTING)
            glDepthMask(GL_FALSE)
            glLineWidth(1.0)
            glColor4f(0.75, 0.9, 1.0, 0.35)
            glBegin(GL_LINES)
            for z0 in (-0.15, 0.0, 0.15):
                for _ in range(25):
                    tx = GL_C + self.fx.uniform(-0.18, 0.18)
                    tz = z0 + self.fx.uniform(-0.12, 0.12)
                    glVertex3f(GL_C, 0.56, z0)
                    glVertex3f(tx, 0.02, tz)
            glEnd()
            glDepthMask(GL_TRUE)
            glEnable(GL_LIGHTING)

    def render(self, view, line, w, h):
        glViewport(0, 0, w, h)
        glClearColor(*view.clear, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(view.fov, w / float(h), 0.1, 80.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*view.eye(), *view.target, 0, 1, 0)
        mv = np.array(glGetDoublev(GL_MODELVIEW_MATRIX), dtype=np.float64).reshape(4, 4).T
        pr = np.array(glGetDoublev(GL_PROJECTION_MATRIX), dtype=np.float64).reshape(4, 4).T

        k = view.light_intensity
        glLightfv(GL_LIGHT0, GL_POSITION, (*view.light, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.25 * k, 0.25 * k, 0.27 * k, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.95 * k, 0.95 * k, 0.9 * k, 1))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (k, k, k, 1))

        self.draw_static(view, line)
        self.draw_shadows(line, view.light)
        self.draw_items(line)
        self.draw_translucent(line)
        return mv, pr


def class_names(mode):
    if mode == "kind":
        return CLASSES
    if mode == "both":
        return [f"{s}_{k}" for s in STAGES for k in CLASSES]
    return STAGES


def label_class(it, mode):
    if mode == "kind":
        return CLASSES.index(it.kind)
    if mode == "both":
        return it.stage() * len(CLASSES) + CLASSES.index(it.kind)
    return it.stage()


def grab_frame(w, h):
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    glReadBuffer(GL_BACK)
    data = glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE)
    img = data.reshape(h, w, 3) if isinstance(data, np.ndarray) \
        else np.frombuffer(data, np.uint8).reshape(h, w, 3)
    return np.ascontiguousarray(img[::-1])


def project(points, mv, pr, w, h):
    ph = np.hstack([points, np.ones((len(points), 1))])
    clip = ph @ (pr @ mv).T
    if np.any(clip[:, 3] <= 1e-6):
        return None
    ndc = clip[:, :3] / clip[:, 3:4]
    return (ndc[:, 0] * 0.5 + 0.5) * w, (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h


def make_labels(line, renderer, mv, pr, w, h, mode, min_visible=0.6):
    labels = []
    for it in line.items:
        mesh = renderer.get_mesh(it.kind, it.level())
        res = project(it.world_points(mesh), mv, pr, w, h)
        if res is None:
            continue
        xs, ys = res
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        full = (x1 - x0) * (y1 - y0)
        cx0, cx1, cy0, cy1 = max(x0, 0), min(x1, w), max(y0, 0), min(y1, h)
        if full <= 0 or cx1 <= cx0 or cy1 <= cy0:
            continue
        if (cx1 - cx0) * (cy1 - cy0) / full < min_visible:
            continue
        if (cx1 - cx0) < 6 or (cy1 - cy0) < 6:
            continue
        labels.append((label_class(it, mode), (cx0 + cx1) / 2 / w, (cy0 + cy1) / 2 / h,
                       (cx1 - cx0) / w, (cy1 - cy0) / h))
    return labels


def augment(img, rng):
    out = img.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
    sigma = rng.uniform(0, 7)
    if sigma > 0.5:
        out += np.random.normal(0, sigma, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    k = rng.choice([1, 1, 3, 5])
    return cv2.GaussianBlur(out, (k, k), 0) if k > 1 else out


class DatasetWriter:
    def __init__(self, root, names, preview=False):
        self.root, self.names, self.preview = root, names, preview
        self.img_dir = os.path.join(root, "images")
        self.lbl_dir = os.path.join(root, "labels")
        self.prev_dir = os.path.join(root, "preview")
        for d in [self.img_dir, self.lbl_dir] + ([self.prev_dir] if preview else []):
            os.makedirs(d, exist_ok=True)
        self.count = len([f for f in os.listdir(self.img_dir) if f.endswith(".jpg")])
        self.stats = Counter()
        with open(os.path.join(root, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(names))
        with open(os.path.join(root, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(f"path: {os.path.abspath(root)}\ntrain: images\nval: images\nnames:\n"
                    + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))

    def save(self, img, labels):
        if not labels:
            return False
        self.count += 1
        name = f"{self.count:06d}"
        cv2.imwrite(os.path.join(self.img_dir, name + ".jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(self.lbl_dir, name + ".txt"), "w") as f:
            for c, x, y, bw, bh in labels:
                f.write(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")
                self.stats[self.names[c]] += 1
        if self.preview:
            h, w = img.shape[:2]
            vis = img.copy()
            for c, x, y, bw, bh in labels:
                p1 = (int((x - bw / 2) * w), int((y - bh / 2) * h))
                p2 = (int((x + bw / 2) * w), int((y + bh / 2) * h))
                color = (0, 0, 255) if self.names[c].startswith("defect") else (0, 255, 0)
                cv2.rectangle(vis, p1, p2, color, 2)
                cv2.putText(vis, self.names[c], (p1[0], max(p1[1] - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(self.prev_dir, name + ".jpg"), vis)
        return True

    def finish(self):
        with open(os.path.join(self.root, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(dict(self.stats), f, ensure_ascii=False, indent=2)
        print("Розподіл об'єктів за класами:", dict(self.stats))


def randomize_view(view, rng):
    if rng.random() < 0.7:      
        sx = rng.choice(list(STATION_X.values()))
        view.target = np.array([sx + rng.uniform(-0.8, 0.8), 0.0, 0.0])
        view.dist = rng.uniform(1.8, 3.6)
    else:                   
        view.target = np.array([rng.uniform(-4, 4), 0.0, 0.0])
        view.dist = rng.uniform(4.0, 8.0)
    view.yaw = rng.uniform(65, 115)
    view.pitch = rng.uniform(25, 65)
    view.fov = rng.uniform(38, 55)
    view.light = np.array([rng.uniform(-2, 2) + view.target[0], rng.uniform(2.0, 4.5),
                           rng.uniform(-2, 2)])
    view.light_intensity = rng.uniform(0.7, 1.2)
    view.belt_shade = rng.uniform(0.08, 0.35)
    g = rng.uniform(0.35, 0.8)
    view.clear = (g, g * rng.uniform(0.95, 1.05), g * rng.uniform(0.95, 1.1))


class App:
    def __init__(self, args):
        self.args = args
        self.rng = random.Random(args.seed)
        np.random.seed(args.seed)
        self.view = View()
        self.paused = self.recording = self.save_one = self.drag = False
        self.last = (0, 0)
        self.names = class_names(args.labels)

    def init_window(self):
        if not glfw.init():
            sys.exit("Не вдалося ініціалізувати GLFW")
        glfw.window_hint(glfw.SAMPLES, 4)
        if self.args.hidden or self.args.dataset > 0:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.win = glfw.create_window(self.args.width, self.args.height,
                                      "Lab9 - Dishware production line", None, None)
        if not self.win:
            glfw.terminate()
            sys.exit("Не вдалося створити вікно OpenGL")
        glfw.make_context_current(self.win)
        glfw.swap_interval(0 if self.args.dataset > 0 else 1)
        glfw.set_mouse_button_callback(self.win, self.on_mouse)
        glfw.set_cursor_pos_callback(self.win, self.on_cursor)
        glfw.set_scroll_callback(self.win, self.on_scroll)
        glfw.set_key_callback(self.win, self.on_key)

    def on_mouse(self, win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.drag = action == glfw.PRESS
            self.last = glfw.get_cursor_pos(win)

    def on_cursor(self, win, x, y):
        if self.drag:
            self.view.yaw += (x - self.last[0]) * 0.3
            self.view.pitch = float(np.clip(self.view.pitch + (y - self.last[1]) * 0.3, 5, 89))
        self.last = (x, y)

    def on_scroll(self, win, dx, dy):
        self.view.dist = float(np.clip(self.view.dist * (0.9 ** dy), 0.8, 20.0))

    def on_key(self, win, key, sc, action, mods):
        if action == glfw.RELEASE:
            return
        v = self.view
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            self.paused = not self.paused
        elif key == glfw.KEY_R and action == glfw.PRESS:
            self.recording = not self.recording
            print("Запис датасету:", "ON" if self.recording else "OFF")
        elif key == glfw.KEY_S and action == glfw.PRESS:
            self.save_one = True
        elif key == glfw.KEY_0:
            v.target, v.dist, v.pitch, v.yaw = np.array([0.0, 0, 0]), 10.0, 30.0, 90.0
        elif glfw.KEY_1 <= key <= glfw.KEY_7:
            n = key - glfw.KEY_0
            v.target = np.array([STATION_X[n], 0.0, 0.0])
            v.dist, v.pitch, v.yaw = 2.4, 30.0, 90.0
            print("Станція:", STATION_NAMES[n])
        elif key == glfw.KEY_LEFT:
            v.light[0] -= 0.1
        elif key == glfw.KEY_RIGHT:
            v.light[0] += 0.1
        elif key == glfw.KEY_UP:
            v.light[2] -= 0.1
        elif key == glfw.KEY_DOWN:
            v.light[2] += 0.1
        elif key == glfw.KEY_PAGE_UP:
            v.light[1] += 0.1
        elif key == glfw.KEY_PAGE_DOWN:
            v.light[1] = max(0.5, v.light[1] - 0.1)

    def capture(self, writer, line, renderer, mv, pr, augment_img):
        w, h = glfw.get_framebuffer_size(self.win)
        img = grab_frame(w, h)
        labels = make_labels(line, renderer, mv, pr, w, h, self.args.labels)
        if augment_img:
            img = augment(img, self.rng)
        return writer.save(img, labels)

    def run(self):
        self.init_window()
        renderer = Renderer()
        renderer.init_gl()
        line = Line(self.rng)
        for _ in range(900):                   
            line.step(SIM_DT)
        writer = DatasetWriter(self.args.out, self.names, self.args.preview)
        if self.args.dataset > 0:
            self.run_dataset(renderer, line, writer)
        else:
            self.run_interactive(renderer, line, writer)
        writer.finish()
        glfw.terminate()

    def run_dataset(self, renderer, line, writer):
        saved, target = 0, self.args.dataset
        while saved < target and not glfw.window_should_close(self.win):
            for _ in range(self.args.every):
                line.step(SIM_DT)
            randomize_view(self.view, self.rng)
            w, h = glfw.get_framebuffer_size(self.win)
            mv, pr = renderer.render(self.view, line, w, h)
            if self.capture(writer, line, renderer, mv, pr, True):
                saved += 1
                if saved % 25 == 0 or saved == target:
                    print(f"[{saved}/{target}] кадрів у '{self.args.out}'")
            glfw.swap_buffers(self.win)
            glfw.poll_events()

    def run_interactive(self, renderer, line, writer):
        print("Станції: " + "; ".join(f"{k}-{v}" for k, v in STATION_NAMES.items()))
        acc, prev, step_no, t_title = 0.0, glfw.get_time(), 0, 0.0
        while not glfw.window_should_close(self.win):
            now = glfw.get_time()
            acc += min(now - prev, 0.25)
            prev = now
            while acc >= SIM_DT:
                if not self.paused:
                    line.step(SIM_DT)
                    step_no += 1
                acc -= SIM_DT
            w, h = glfw.get_framebuffer_size(self.win)
            mv, pr = renderer.render(self.view, line, w, h)
            if self.save_one or (self.recording and not self.paused
                                 and step_no % self.args.every == 0):
                writer_ok = self.capture(writer, line, renderer, mv, pr, False)
                self.save_one = False
                if writer_ok:
                    t_title = 0
            if now - t_title > 0.5:
                total = line.packed + line.rejected
                yld = 100.0 * line.packed / total if total else 100.0
                glfw.set_window_title(
                    self.win, f"Лінія | у тару: {line.packed} | брак: {line.rejected} | "
                              f"вихід придатних: {yld:.0f}% | кадрів: {writer.count}")
                t_title = now
            glfw.swap_buffers(self.win)
            glfw.poll_events()


def parse_args():
    p = argparse.ArgumentParser(description="Lab 9 (variant 1): dishware production line")
    p.add_argument("--dataset", type=int, default=0, help="к-сть кадрів (0 = інтерактив)")
    p.add_argument("--out", default="dataset_dishware")
    p.add_argument("--labels", choices=["stage", "kind", "both"], default="stage")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--every", type=int, default=6, help="кроків симуляції між кадрами")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", action="store_true")
    p.add_argument("--preview", action="store_true", help="зберігати кадри з рамками")
    return p.parse_args()


if __name__ == "__main__":
    App(parse_args()).run()