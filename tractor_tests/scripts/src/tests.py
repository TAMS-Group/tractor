#!/usr/bin/env python

import time
import math
import numpy as np
import importlib
import tractor as t
import tractor as tr
import tractor
import inspect
import tractor.types_double as tt

def var_vec3_zz(tg):
    v = tg.Vector3()
    t.variable(v)
    return v

def var_vec3_nn(tg):
    v = tg.Vector3()
    t.variable(v)
    v += tg.Vector3(0.001,0,0)
    return v

def var_vec3_nz(tg):
    v = tg.Vector3()
    t.variable(v)
    if tg == tractor.types_double_scalar:
        v += tg.Vector3(1e-9,0,0)
    return v

class Tests:

    def tensor(self, tg):
        ta = tt.Tensor(np.zeros([10,3]))
        tb = tt.Tensor(np.zeros([3,5]))
        tr.variable(ta)
        tr.variable(tb)
        tc = tr.matmul(ta, tb)
        td = tt.Tensor(np.zeros([10,5]))
        tr.variable(td)
        tc += td
        tr.goal(tc)
        for v in tr.unpack(tc):
            tr.goal(v)

    def sinc(self, tg):
        v = tt.Scalar()
        t.variable(v)
        t.goal(t.sinc(v))

    def twist_unpack(self, tg):
        twist = tg.Twist()
        t.variable(twist)
        p, r = t.unpack(twist)
        t.goal(p)
        t.goal(r)

    def twist_scale(self, tg):
        v = tg.Twist()
        t.variable(v)
        f = tt.Scalar()
        t.variable(f)
        t.goal(v * f)
        t.goal(f * v)

    def pose_unpack(self, tg):
        twist = tg.Twist()
        t.variable(twist)
        twist += tg.Twist(tg.Vector3(0,0,0), tg.Vector3(0.001,0,0))
        pose = tg.Pose.identity + twist
        p, r = t.unpack(pose)
        t.goal(p)
        t.goal(t.residual(r))

    def pose_translation(self, tg):
        pos = tg.Vector3()
        t.variable(pos)
        xyzw = [tt.Scalar(1) for i in range(4)]
        for v in xyzw: t.variable(v)
        #q = tg.Orientation(xyzw)
        q = tg.Orientation([v * tt.Scalar(10) + tt.Scalar(0.01) for v in xyzw])
        pose = tg.Pose(pos, q)
        t.goal(t.translation(pose))

    def pose_orientation(self, tg):
        pos = tg.Vector3()
        t.variable(pos)
        xyzw = [tt.Scalar(1) for i in range(4)]
        for v in xyzw: t.variable(v)
        #q = tg.Orientation(xyzw)
        q = tg.Orientation([v * tt.Scalar(10) + tt.Scalar(0.01) for v in xyzw])
        pose = tg.Pose(pos, q)
        t.goal(t.residual(t.orientation(pose)))

    def pose_residual(self, tg):
        pos = tg.Vector3()
        t.variable(pos)
        xyzw = [tt.Scalar(1) for i in range(4)]
        for v in xyzw: t.variable(v)
        #q = tg.Orientation(xyzw)
        q = tg.Orientation([v * tt.Scalar(10) + tt.Scalar(0.01) for v in xyzw])
        pose = tg.Pose(pos, q)
        t.goal(t.residual(pose))

    def pose_residual_2(self, tg):
        twist = tg.Twist()
        t.variable(twist)
        pose = tg.Pose.identity + twist
        t.goal(t.residual(pose))

    def quat_residual(self, tg):

        angle = tt.Scalar()
        t.variable(angle)
        angle *= tt.Scalar(10)

        axis = t.normalized(tg.Vector3(1,6,-3))

        t.goal(t.residual(tg.Orientation.angle_axis(angle, axis + tg.Vector3(0.0001,0,0))))

    def quat_residual_2(self, tg):

        angle_1 = tt.Scalar()
        t.variable(angle_1)
        axis_1 = t.normalized(tg.Vector3(1,2,3))

        angle_2 = tt.Scalar()
        t.variable(angle_2)
        axis_2 = t.normalized(tg.Vector3(4,-5,6))

        angle_1 *= tt.Scalar(0.2)
        angle_2 *= tt.Scalar(0.2)

        t.goal(t.residual(tg.Orientation.angle_axis(angle_1, axis_1), tg.Orientation.angle_axis(angle_2, axis_2)))

    def quat_residual_3(self, tg):

        angle = tt.Scalar()
        t.variable(angle)

        axis = tg.Vector3()
        t.variable(axis)

        t.goal(t.residual(tg.Orientation.angle_axis(angle, axis + tg.Vector3(0.0001,0,0))))

    def quat_pack_residual(self, tg):
        xyzw = [tt.Scalar() for i in range(4)]
        for v in xyzw: t.variable(v)
        t.goal(t.residual(tg.Orientation([v * tt.Scalar(10) + tt.Scalar(0.001) for v in xyzw])))

    def make_pose(self, tg):
        pos = tg.Vector3()
        t.variable(pos)
        xyzw = [tt.Scalar(1) for i in range(4)]
        for v in xyzw: t.variable(v)
        q = tg.Orientation([v + tt.Scalar(0.0001) for v in xyzw])
        pose = tg.Pose(pos, q)
        w = tg.Vector3()
        t.variable(w)
        t.goal(pose * w)

    def pose_twist_add(self, tg):

        angle = tt.Scalar()
        t.variable(angle)

        #axis = tg.Vector3()
        #t.variable(axis)
        axis = var_vec3_nn(tg)

        quat = tg.Orientation.angle_axis(angle, axis + tg.Vector3(0.0001,0,0))

        pos = tg.Vector3()
        t.variable(pos)

        pose = tg.Pose(pos, quat)

        #twist_t = tg.Vector3([1,2,3])
        #twist_r = tg.Vector3([2,3,4])
        #t.variable(twist_t)
        #t.variable(twist_r)

        twist_t = var_vec3_zz(tg)
        twist_r = var_vec3_nn(tg)

        twist = tg.Twist(twist_t, twist_r) * tt.Scalar(10)

        w = tg.Vector3()
        t.variable(w)

        t.goal(t.inverse(pose + twist + twist) * w)

    def quat_mul(self, tg):

        angle_1 = tt.Scalar()
        t.variable(angle_1)
        axis_1 = t.normalized(tg.Vector3(1,2,3))

        angle_2 = tt.Scalar()
        t.variable(angle_2)
        axis_2 = t.normalized(tg.Vector3(-4,5,6))

        vec = tg.Vector3(2,3,4)
        t.variable(vec)
        t.goal((tg.Orientation.angle_axis(angle_1, axis_1) * tg.Orientation.angle_axis(angle_2, axis_2)) * vec)

    def quat_mul_inv(self, tg):

        angle_1 = tt.Scalar()
        t.variable(angle_1)
        axis_1 = t.normalized(tg.Vector3(1,2,3))

        angle_2 = tt.Scalar()
        t.variable(angle_2)
        axis_2 = t.normalized(tg.Vector3(4,7,-2))

        vec = tg.Vector3(2,3,4)
        t.variable(vec)

        t.goal((t.inverse(tg.Orientation.angle_axis(angle_1, axis_1)) * tg.Orientation.angle_axis(angle_2, axis_2)) * vec)
        t.goal((t.inverse(tg.Orientation.angle_axis(angle_1, axis_1)) * t.inverse(tg.Orientation.angle_axis(angle_2, axis_2))) * vec)
        t.goal((tg.Orientation.angle_axis(angle_1, axis_1) * t.inverse(tg.Orientation.angle_axis(angle_2, axis_2))) * vec)

    def vec_to_quat(self, tg):

        #v = tg.Vector3()
        #t.variable(v)
        v = var_vec3_nn(tg)

        q = tg.Orientation().identity + v * tt.Scalar(10)

        w = tg.Vector3()
        t.variable(w)

        t.goal(q * w)

    def vec_to_quat_1(self, tg):

        #v = tg.Vector3()
        #t.variable(v)
        v = var_vec3_nn(tg)

        q = tg.Orientation.identity + v

        w = tg.Vector3()
        t.variable(w)

        t.goal(q * w)

    def vec_to_quat_2(self, tg):

        angle = tt.Scalar()
        t.variable(angle)

        #axis = tg.Vector3()
        #t.variable(axis)
        axis = var_vec3_nn(tg)

        #w = tg.Vector3()
        #t.variable(w)
        w = var_vec3_nn(tg)

        q = tg.Orientation.angle_axis(angle, axis) + w

        t.goal(q * tg.Vector3(1,2,3))

    def vec_to_quat_3(self, tg):

        angle_1 = tt.Scalar()
        t.variable(angle_1)
        axis_1 = tg.Vector3()
        t.variable(axis_1)
        axis_1 += tg.Vector3(0.01,0,0)

        angle_2 = tt.Scalar()
        t.variable(angle_2)
        axis_2 = tg.Vector3()
        t.variable(axis_2)
        axis_2 += tg.Vector3(0.01,0,0)

        q = tg.Orientation.angle_axis(angle_1, axis_1) * tg.Orientation.angle_axis(angle_2, axis_2) + tg.Vector3(-1,3,2)

        w = tg.Vector3()
        t.variable(w)

        t.goal(q * w)

    def vec_to_quat_inv(self, tg):

        angle = tt.Scalar()
        t.variable(angle)

        #axis = tg.Vector3()
        #t.variable(axis)
        #axis += tg.Vector3(0.01,0,0)
        axis = var_vec3_nn(tg)

        w = var_vec3_nn(tg)
        #w = tg.Vector3()
        #t.variable(w)
        #w += tg.Vector3(0.01,0,0)

        q = t.inverse(tg.Orientation.angle_axis(angle, axis)) + w

        t.goal(t.inverse(q) * tg.Vector3(1,2,3))

    def twist_translation(self, tg):
        v = tg.Vector3()
        t.variable(v)
        x = tg.Twist(v, tg.Vector3.zero)
        t.goal(x)

    def quat_angle_axis_va_unpack(self, tg):
        a = tt.Scalar(1)
        t.variable(a)
        v = var_vec3_nn(tg)
        q = tg.Orientation.angle_axis(a, v)
        q = t.unpack(q)
        for v in q:
            t.goal(v)

    def quat_angle_axis_va(self, tg):
        a = tt.Scalar(1)
        v = var_vec3_nn(tg)
        w = tg.Vector3([2,3,4])
        t.variable(a)
        t.variable(w)
        t.goal(tg.Orientation.angle_axis(a, v) * w)

    def quat_angle_axis_va_inv(self, tg):
        a = tt.Scalar(1)
        v = var_vec3_nn(tg)
        w = tg.Vector3([2,3,4])
        t.variable(a)
        t.variable(w)
        t.goal(t.inverse(tg.Orientation.angle_axis(a, v)) * w)

    def quat_angle_axis_v(self, tg):
        axis = var_vec3_nn(tg)
        angle = tt.Scalar(1)
        testvector = tg.Vector3(0,1,0)
        t.goal(tg.Orientation.angle_axis(angle, t.normalized(axis)) * testvector - testvector)

    def quat_unpack(self, tg):

        a1 = tt.Scalar()
        t.variable(a1)
        q1 = tg.Orientation.angle_axis(a1, t.normalized(tg.Vector3(1,2,3)))

        a2 = tt.Scalar()
        t.variable(a2)
        q2 = tg.Orientation.angle_axis(a2, t.normalized(tg.Vector3(-4,3,1)))

        xyzw = t.unpack(q1 * q2)
        for v in xyzw:
            t.goal(v)

    def quat_pack(self, tg):

        a1 = tt.Scalar()
        t.variable(a1)
        q1 = tg.Orientation.angle_axis(a1, t.normalized(tg.Vector3(1,2,3)))

        a2 = tt.Scalar()
        t.variable(a2)
        q2 = tg.Orientation.angle_axis(a2, t.normalized(tg.Vector3(-4,3,1)))

        xyzw = t.unpack(q1 * q2)

        w = tg.Vector3()
        t.variable(w)

        t.goal(tg.Orientation(xyzw) * w);

    def quat_pack_2(self, tg):
        xyzw = [tt.Scalar() for i in range(4)]
        for v in xyzw: t.variable(v)
        w = tg.Vector3()
        t.variable(w)
        t.goal(tg.Orientation([v + tt.Scalar(0.0001) for v in xyzw]) * w)

    def quat_angle_axis_a(self, tg):
        a = tt.Scalar()
        w = tg.Vector3()
        t.variable(a)
        t.variable(w)
        v = t.normalized(tg.Vector3(1,2,3))
        t.goal(tg.Orientation.angle_axis(a, v) * w)

    def pose_angle_axis_a(self, tg):
        a = tt.Scalar()
        w = tg.Vector3()
        t.variable(a)
        t.variable(w)
        v = t.normalized(tg.Vector3(1,2,3))
        t.goal(tg.Pose.angle_axis(a, v) * w)

    def pose_angle_axis_va(self, tg):
        a = tt.Scalar()
        v = var_vec3_nn(tg)
        w = tg.Vector3()
        t.variable(a)
        t.variable(w)
        t.goal(tg.Pose.angle_axis(a, v) * w)

    def pose_angle_axis_p(self, tg):

        a = tt.Scalar()
        v = var_vec3_nn(tg)
        t.variable(a)
        pa = tg.Pose.angle_axis(a, v)

        a = tt.Scalar()
        v = var_vec3_nn(tg)
        t.variable(a)
        pb = tg.Pose.angle_axis(pa, a, v)

        a = tt.Scalar()
        v = var_vec3_nn(tg)
        t.variable(a)
        pc = tg.Pose.angle_axis(pb, a, v)

        a = tt.Scalar()
        v = var_vec3_nn(tg)
        t.variable(a)
        pd = t.inverse(tg.Pose.angle_axis(a, v)) * pa

        w = tg.Vector3()
        t.variable(w)

        t.goal(pd * w)

    def pose_angle_axis_inverse(self, tg):

        a = tt.Scalar()
        v = var_vec3_nn(tg)
        t.variable(a)
        p = t.inverse(tg.Pose.angle_axis(a * tt.Scalar(10), v))

        w = tg.Vector3()
        t.variable(w)

        t.goal(p * w)

    def vec3_scale(self, tg):
        v = tg.Vector3()
        t.variable(v)
        f = tt.Scalar()
        t.variable(f)
        t.goal(v * f)
        t.goal(f * v)

    def vec3_unpack(self, tg):
        a = tg.Vector3()
        t.variable(a)
        data = t.unpack(a)
        for d in data:
            t.goal(d)

    def vec3_pack(self, tg):
        a = tt.Scalar()
        b = tt.Scalar()
        c = tt.Scalar()
        t.variable(a)
        t.variable(b)
        t.variable(c)
        t.goal(tg.Vector3(a, b, c))

    def vec3_cross(self, tg):
        a = tg.Vector3()
        b = tg.Vector3()
        t.variable(a)
        t.variable(b)
        t.goal(t.cross(a, b))

    def vec3_norms(self, tg):
       a = tg.Vector3()
       t.variable(a)
       t.goal(t.norm(a + tg.Vector3(0.1,0,0)))
       t.goal(t.squaredNorm(a))
       t.goal(t.normalized(a + tg.Vector3(0.1,0,0)))

    def vec3_normalize(self, tg):
       a = tg.Vector3()
       t.variable(a)
       a += tg.Vector3(0.1,0,0)
       t.goal(t.normalized(a))

    def vec3_dot(self, tg):
        a = tg.Vector3()
        b = tg.Vector3()
        t.variable(a)
        t.variable(b)
        t.goal(t.dot(a, b))

    def mat3(self, tg):
        vec = tg.Vector3()
        mat1 = tg.Matrix3()
        mat2 = tg.Matrix3()
        mat3 = tg.Matrix3()
        t.variable(vec)
        t.variable(mat1)
        t.variable(mat2)
        t.variable(mat3)
        vec = (mat1 + -mat2 - mat3) * -vec
        t.goal(vec)

    def mat3_inverse(self, tg):
        m = tg.Matrix3()
        t.variable(m)
        t.goal(t.inverse(m + tg.Matrix3.identity + tg.Matrix3.identity))

    def stuff(self, tg):
        vec = tg.Vector3()
        t.variable(vec)
        angle = tt.Scalar()
        t.variable(angle)
        pose = tg.Pose.angle_axis(angle, tg.Vector3(1,2,3) * tt.Scalar(1/math.sqrt(1*1+2*2+3*3)))
        angle = tt.Scalar()
        t.variable(angle)
        pose2 = tg.Pose.angle_axis(angle, tg.Vector3(1,0,0))
        for i in range(3):
            vec = pose * pose2 * t.inverse(pose * pose) * vec + vec
        angle = tt.Scalar()
        t.variable(angle)
        orientation = tg.Orientation.angle_axis(angle, tg.Vector3(1,2,3) * tt.Scalar(1/math.sqrt(1*1+2*2+3*3)))
        for i in range(2):
            vec = orientation * vec + vec
        t.goal(vec)

    def sin_cos(self, tg):
        a = tt.Scalar()
        t.variable(a)
        t.goal(t.sin(a))
        t.goal(t.cos(a))
        s = tt.Scalar()
        c = tt.Scalar()
        t.sincos(a, s, c)
        t.variable(s)
        t.variable(c)

    def exp(self, tg):
        a = tt.Scalar()
        t.variable(a)
        t.goal(t.exp(a))

    def log(self, tg):
        a = tt.Scalar()
        t.variable(a)
        t.goal(t.log(a * a + tt.Scalar(0.1)))

    def sqrt(self, tg):
        a = tt.Scalar()
        t.variable(a)
        t.goal(t.sqrt(a * a + tt.Scalar(0.001)))

    def tanh(self, tg):
        a = tt.Scalar()
        t.variable(a)
        t.goal(t.tanh(a))

    # def test_fail(self, tg):
    #     a = tt.Scalar()
    #     b = tt.Scalar()
    #     t.variable(a)
    #     t.variable(b)
    #     t.goal(t.add_random_normal(a, b))

    # def fail(self, tg):
    #     a = tt.Scalar()
    #     b = tt.Scalar()
    #     t.variable(a)
    #     t.variable(b)
    #     x = t.add_random_normal(a, b)
    #     t.goal(x)
